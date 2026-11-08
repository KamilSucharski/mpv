/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <libswscale/swscale.h>

#include "common/msg.h"
#include "input/input.h"
#include "options/options.h"
#include "osdep/timer.h"
#include "present_sync.h"
#include "sub/draw_bmp.h"
#include "sub/osd.h"
#include "video/csputils.h"
#include "video/fmt-conversion.h"
#include "video/mp_image.h"
#include "video/sws_utils.h"
#include "vo.h"
#include "x11_common.h"

// Driver context structure
struct x11vo_priv {
    struct vo *parent_vo;
    struct mp_image *last_frame;

    XImage *framebuf[2];
    struct mp_image wrapped_imgs[2];
    int pixel_depth;
    GC gfx_context;

    uint32_t buf_width;
    uint32_t buf_height;

    struct mp_rect source_crop;
    struct mp_rect dest_region;
    struct mp_osd_res osd_config;

    struct mp_sws_context *sw_scaler;
    XVisualInfo visual_info;
    int front_buffer;

    int using_shm;
    XShmSegmentInfo shm_segments[2];
    int slow_path_warned;
};

// Create mask from component description
#define BUILD_MASK(c) (((1ul << (c).size) - 1) << (c).offset)

static bool reconfigure_buffers(struct vo *vo);

// Create XImage with shared memory or fallback to standard allocation
static bool create_buffer(struct x11vo_priv *priv, int buf_idx)
{
    struct vo *vo = priv->parent_vo;
    Display *disp = vo->x11->display;

    // Probe for MIT-SHM support
    if (vo->x11->display_is_local && XShmQueryExtension(disp)) {
        priv->using_shm = 1;
        vo->x11->ShmCompletionEvent = XShmGetEventBase(disp) + ShmCompletion;
    } else {
        priv->using_shm = 0;
        MP_WARN(vo, "MIT-SHM unavailable, using fallback rendering\n");
    }

    if (priv->using_shm) {
        priv->framebuf[buf_idx] = XShmCreateImage(
            disp, priv->visual_info.visual, priv->pixel_depth,
            ZPixmap, NULL, &priv->shm_segments[buf_idx],
            priv->buf_width, priv->buf_height);

        if (!priv->framebuf[buf_idx]) {
            MP_WARN(vo, "XShmCreateImage failed\n");
            goto standard_alloc;
        }

        priv->shm_segments[buf_idx].shmid = shmget(
            IPC_PRIVATE,
            priv->framebuf[buf_idx]->bytes_per_line *
            priv->framebuf[buf_idx]->height,
            IPC_CREAT | 0777);

        if (priv->shm_segments[buf_idx].shmid < 0) {
            XDestroyImage(priv->framebuf[buf_idx]);
            MP_WARN(vo, "shmget failed for buffer\n");
            goto standard_alloc;
        }

        priv->shm_segments[buf_idx].shmaddr =
            shmat(priv->shm_segments[buf_idx].shmid, 0, 0);

        if (priv->shm_segments[buf_idx].shmaddr == (char *)-1) {
            XDestroyImage(priv->framebuf[buf_idx]);
            MP_WARN(vo, "shmat failed for buffer\n");
            goto standard_alloc;
        }

        priv->framebuf[buf_idx]->data = priv->shm_segments[buf_idx].shmaddr;
        priv->shm_segments[buf_idx].readOnly = False;
        XShmAttach(disp, &priv->shm_segments[buf_idx]);
        XSync(disp, False);
        shmctl(priv->shm_segments[buf_idx].shmid, IPC_RMID, 0);
        return true;
    }

standard_alloc:
    priv->using_shm = 0;
    MP_VERBOSE(vo, "Using regular XImage allocation.\n");

    priv->framebuf[buf_idx] = XCreateImage(
        disp, priv->visual_info.visual, priv->pixel_depth,
        ZPixmap, 0, NULL, priv->buf_width, priv->buf_height, 8, 0);

    if (priv->framebuf[buf_idx]) {
        size_t alloc_size = priv->framebuf[buf_idx]->bytes_per_line *
                            priv->buf_height + 32;
        priv->framebuf[buf_idx]->data = calloc(1, alloc_size);
    }

    if (!priv->framebuf[buf_idx] || !priv->framebuf[buf_idx]->data) {
        MP_WARN(vo, "XImage allocation failed\n");
        return false;
    }
    return true;
}

// Release XImage buffer resources
static void destroy_buffer(struct x11vo_priv *priv, int buf_idx)
{
    struct vo *vo = priv->parent_vo;

    if (!priv->framebuf[buf_idx])
        return;

    if (priv->using_shm) {
        XShmDetach(vo->x11->display, &priv->shm_segments[buf_idx]);
        XDestroyImage(priv->framebuf[buf_idx]);
        shmdt(priv->shm_segments[buf_idx].shmaddr);
    } else {
        free(priv->framebuf[buf_idx]->data);
        priv->framebuf[buf_idx]->data = NULL;
        XDestroyImage(priv->framebuf[buf_idx]);
    }
    priv->framebuf[buf_idx] = NULL;
}

// Handle video parameters reconfiguration
static int handle_reconfig(struct vo *vo, struct mp_image_params *params)
{
    vo_x11_config_vo_window(vo);
    return reconfigure_buffers(vo) ? 0 : -1;
}

// Resize and reconfigure internal buffers
static bool reconfigure_buffers(struct vo *vo)
{
    struct x11vo_priv *priv = vo->priv;

    int aligned_w = MPMAX(1, MP_ALIGN_UP(vo->dwidth, MP_IMAGE_BYTE_ALIGN));
    int aligned_h = MPMAX(1, vo->dheight);

    // Grow buffers if needed
    if (aligned_w > priv->buf_width || aligned_h > priv->buf_height) {
        destroy_buffer(priv, 0);
        destroy_buffer(priv, 1);

        priv->buf_width = aligned_w;
        priv->buf_height = aligned_h;

        for (int idx = 0; idx < 2; idx++) {
            if (!create_buffer(priv, idx)) {
                priv->buf_width = 0;
                priv->buf_height = 0;
                return false;
            }
        }
    }

    // Determine matching mpv pixel format
    int selected_fmt = 0;
    for (int fmt = IMGFMT_START; fmt < IMGFMT_END; fmt++) {
        struct mp_imgfmt_desc fdesc = mp_imgfmt_get_desc(fmt);

        bool is_rgb_packed = (fdesc.flags & MP_IMGFLAG_HAS_COMPS) &&
                             fdesc.num_planes == 1 &&
                             (fdesc.flags & MP_IMGFLAG_COLOR_MASK) == MP_IMGFLAG_COLOR_RGB &&
                             (fdesc.flags & MP_IMGFLAG_TYPE_MASK) == MP_IMGFLAG_TYPE_UINT &&
                             (fdesc.flags & MP_IMGFLAG_NE) &&
                             !(fdesc.flags & MP_IMGFLAG_ALPHA);

        if (!is_rgb_packed)
            continue;
        if (fdesc.bpp[0] > 8 * sizeof(unsigned long))
            continue;
        if (priv->framebuf[0]->bits_per_pixel != fdesc.bpp[0])
            continue;
        if (priv->framebuf[0]->byte_order != MP_SELECT_LE_BE(LSBFirst, MSBFirst))
            continue;

        struct mp_imgfmt_desc work = fdesc;

        // Adjust for big endian bit ordering
        if (MP_SELECT_LE_BE(0, 1)) {
            if (!mp_find_other_endian(fmt)) {
                for (int comp = 0; comp < 3; comp++) {
                    work.comps[comp].offset =
                        fdesc.bpp[0] - fdesc.comps[comp].size - fdesc.comps[comp].offset;
                }
            }
        }

        if (priv->framebuf[0]->red_mask == BUILD_MASK(work.comps[0]) &&
            priv->framebuf[0]->green_mask == BUILD_MASK(work.comps[1]) &&
            priv->framebuf[0]->blue_mask == BUILD_MASK(work.comps[2]))
        {
            selected_fmt = fmt;
            break;
        }
    }

    if (!selected_fmt) {
        MP_ERR(vo, "Server pixel format not compatible. Try another VO.\n");
        return false;
    }

    MP_VERBOSE(vo, "Selected format: %s\n", mp_imgfmt_to_name(selected_fmt));

    // Wrap XImage data in mp_image structures
    for (int idx = 0; idx < 2; idx++) {
        struct mp_image *wrapper = &priv->wrapped_imgs[idx];
        *wrapper = (struct mp_image){0};
        mp_image_setfmt(wrapper, selected_fmt);
        mp_image_set_size(wrapper, priv->buf_width, priv->buf_height);
        wrapper->planes[0] = priv->framebuf[idx]->data;
        wrapper->stride[0] = priv->framebuf[idx]->bytes_per_line;
        mp_image_params_guess_csp(&wrapper->params);
    }

    vo_get_src_dst_rects(vo, &priv->source_crop, &priv->dest_region, &priv->osd_config);

    // Setup software scaler
    if (vo->params) {
        priv->sw_scaler->src = *vo->params;
        priv->sw_scaler->src.w = mp_rect_w(priv->source_crop);
        priv->sw_scaler->src.h = mp_rect_h(priv->source_crop);

        priv->sw_scaler->dst = priv->wrapped_imgs[0].params;
        priv->sw_scaler->dst.w = mp_rect_w(priv->dest_region);
        priv->sw_scaler->dst.h = mp_rect_h(priv->dest_region);

        if (mp_sws_reinit(priv->sw_scaler) < 0)
            return false;

        mp_mutex_lock(&vo->params_mutex);
        vo->target_params = &priv->sw_scaler->dst;
        mp_mutex_unlock(&vo->params_mutex);
    }

    vo->want_redraw = true;
    return true;
}

// Send current buffer contents to X server
static void output_buffer(struct x11vo_priv *priv)
{
    struct vo *vo = priv->parent_vo;
    XImage *img = priv->framebuf[priv->front_buffer];

    if (priv->using_shm) {
        XShmPutImage(vo->x11->display, vo->x11->window, priv->gfx_context,
                     img, 0, 0, 0, 0, vo->dwidth, vo->dheight, True);
        vo->x11->ShmCompletionWaitCount++;
    } else {
        XPutImage(vo->x11->display, vo->x11->window, priv->gfx_context,
                  img, 0, 0, 0, 0, vo->dwidth, vo->dheight);
    }
}

// Block until SHM operations complete
static void await_shm_sync(struct vo *vo, int max_pending)
{
    struct x11vo_priv *priv = vo->priv;
    struct vo_x11_state *xstate = vo->x11;

    if (!priv->using_shm)
        return;

    while (xstate->ShmCompletionWaitCount > max_pending) {
        if (!priv->slow_path_warned) {
            MP_WARN(vo, "Rendering too slow, waiting for SHM sync...\n");
            priv->slow_path_warned = 1;
        }
        mp_sleep_ns(MP_TIME_MS_TO_NS(1));
        vo_x11_check_events(vo);
    }
}

// Present the rendered frame
static void present_frame(struct vo *vo)
{
    struct x11vo_priv *priv = vo->priv;

    output_buffer(priv);
    priv->front_buffer = (priv->front_buffer + 1) % 2;

    if (vo->x11->use_present) {
        vo_x11_present(vo);
        present_sync_swap(vo->x11->present);
    }
}

// Retrieve vsync information
static void query_vsync(struct vo *vo, struct vo_vsync_info *vsync)
{
    struct vo_x11_state *xstate = vo->x11;
    if (xstate->use_present)
        present_sync_get_info(xstate->present, vsync);
}

// Render a video frame to the buffer
static bool render_video(struct vo *vo, struct vo_frame *frame)
{
    struct x11vo_priv *priv = vo->priv;

    await_shm_sync(vo, 1);

    if (!vo_x11_check_visible(vo))
        return false;

    struct mp_image *dest = &priv->wrapped_imgs[priv->front_buffer];

    if (frame->current) {
        mp_image_clear_rc_inv(dest, priv->dest_region);

        struct mp_image *input = frame->current;
        struct mp_rect crop = priv->source_crop;
        crop.x0 = MP_ALIGN_DOWN(crop.x0, input->fmt.align_x);
        crop.y0 = MP_ALIGN_DOWN(crop.y0, input->fmt.align_y);
        mp_image_crop_rc(input, crop);

        struct mp_image output = *dest;
        mp_image_crop_rc(&output, priv->dest_region);

        mp_sws_scale(priv->sw_scaler, &output, input);
    } else {
        mp_image_clear(dest, 0, 0, dest->w, dest->h);
    }

    double timestamp = frame->current ? frame->current->pts : 0;
    osd_draw_on_image(vo->osd, priv->osd_config, timestamp, 0, dest);

    if (frame->current != priv->last_frame)
        priv->last_frame = frame->current;

    return true;
}

// Check if format is supported
static int format_supported(struct vo *vo, int fmt)
{
    struct x11vo_priv *priv = vo->priv;
    return mp_sws_supports_formats(priv->sw_scaler, IMGFMT_RGB0, fmt) ? 1 : 0;
}

// Cleanup resources
static void driver_uninit(struct vo *vo)
{
    struct x11vo_priv *priv = vo->priv;

    destroy_buffer(priv, 0);
    destroy_buffer(priv, 1);

    if (priv->gfx_context)
        XFreeGC(vo->x11->display, priv->gfx_context);

    vo_x11_uninit(vo);
}

// Initialize the driver
static int driver_init(struct vo *vo)
{
    struct x11vo_priv *priv = vo->priv;
    priv->parent_vo = vo;
    priv->sw_scaler = mp_sws_alloc(vo);
    priv->sw_scaler->log = vo->log;
    mp_sws_enable_cmdline_opts(priv->sw_scaler, vo->global);

    if (!vo_x11_init(vo))
        goto failure;

    struct vo_x11_state *xstate = vo->x11;

    XWindowAttributes root_attrs;
    XGetWindowAttributes(xstate->display, xstate->rootwin, &root_attrs);
    priv->pixel_depth = root_attrs.depth;

    if (!XMatchVisualInfo(xstate->display, xstate->screen, priv->pixel_depth,
                          TrueColor, &priv->visual_info))
        goto failure;

    MP_VERBOSE(vo, "Visual ID: %d\n", (int)priv->visual_info.visualid);

    if (!vo_x11_create_vo_window(vo, &priv->visual_info, "x11"))
        goto failure;

    priv->gfx_context = XCreateGC(xstate->display, xstate->window, 0, NULL);

    MP_WARN(vo, "This VO uses software rendering and has limited performance. "
                "Consider using a GPU-accelerated VO.\n");
    return 0;

failure:
    driver_uninit(vo);
    return -1;
}

// Handle control commands
static int driver_control(struct vo *vo, uint32_t request, void *data)
{
    if (request == VOCTRL_SET_PANSCAN) {
        if (vo->config_ok)
            reconfigure_buffers(vo);
        return VO_TRUE;
    }

    int events = 0;
    int result = vo_x11_control(vo, &events, request, data);

    if (vo->config_ok && (events & (VO_EVENT_EXPOSE | VO_EVENT_RESIZE)))
        reconfigure_buffers(vo);

    vo_event(vo, events);
    return result;
}

const struct vo_driver video_out_x11 = {
    .description = "X11 (software scaling)",
    .name = "x11",
    .priv_size = sizeof(struct x11vo_priv),
    .preinit = driver_init,
    .query_format = format_supported,
    .reconfig = handle_reconfig,
    .control = driver_control,
    .draw_frame = render_video,
    .flip_page = present_frame,
    .get_vsync = query_vsync,
    .wakeup = vo_x11_wakeup,
    .wait_events = vo_x11_wait_events,
    .uninit = driver_uninit,
};
