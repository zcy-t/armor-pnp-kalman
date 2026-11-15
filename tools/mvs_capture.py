# -*- coding: utf-8 -*-
"""
在 VMware 虚拟机（Ubuntu）中通过 MVS SDK 采集工业相机实时视频 / 抓拍。

用途：作业要求「相机采集」，本脚本可直接调用虚拟机内安装的海康 MVS SDK，
      从 MV-CS016-10UC 采集 1440x1080 RGB 连续帧，输出 AVI 到共享目录，
      供 camera_calibration.py / main.py 使用。

前置条件（虚拟机内）：
    /opt/MVS 已安装；相机通过 USB3 直通到虚拟机；
    /opt/MVS/Samples/64/Python/MvImport 存在；
    VMware 共享目录已挂载：
        sudo mount -t fuse.vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other

运行（在虚拟机里）：
    export MVCAM_COMMON_RUNENV=/opt/MVS/lib
    python3 tools/mvs_capture.py --frames 150 \
        --out /mnt/hgfs/mvs_shots/armor/Video_live.avi

注意：MVS 图形客户端（/opt/MVS/bin/MVS）会独占相机。若报 0x80000203
      （MV_E_ACCESS_DENIED），先关闭 MVS 客户端再运行本脚本；采集完成后
      可用 /opt/MVS/bin/MVS.sh 重新启动客户端。
"""
import argparse
import os
import sys
from ctypes import *  # noqa: F403

import cv2
import numpy as np

MVS_ROOT = os.environ.get("MVS_ROOT", "/opt/MVS")
sys.path.append(os.path.join(MVS_ROOT, "Samples", "64", "Python", "MvImport"))

try:
    from MvCameraControl_class import *  # noqa: F403
except ImportError as exc:
    raise SystemExit(
        "未找到 MvCameraControl_class，请确认 MVS SDK 已安装且 "
        "MVCAM_COMMON_RUNENV=/opt/MVS/lib 已设置：%s" % exc) from exc

BAYER = {
    PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerRG2BGR,       # noqa: F405
    PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGB2BGR,       # noqa: F405
    PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGR2BGR,       # noqa: F405
    PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerBG2BGR,       # noqa: F405
}


def to_bgr(buf, w, h, pixel_type):
    if pixel_type == PixelType_Gvsp_RGB8_Packed:          # noqa: F405
        arr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if pixel_type in BAYER:
        arr = np.frombuffer(buf, np.uint8).reshape(h, w)
        return cv2.cvtColor(arr, BAYER[pixel_type])
    if pixel_type == PixelType_Gvsp_Mono8:                # noqa: F405
        arr = np.frombuffer(buf, np.uint8).reshape(h, w)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    raise RuntimeError("不支持的像素格式: 0x%x" % pixel_type)


def main():
    ap = argparse.ArgumentParser(description="MVS SDK 实时采集")
    ap.add_argument("--frames", type=int, default=150, help="采集帧数")
    ap.add_argument("--fps", type=float, default=30.0, help="输出视频帧率")
    ap.add_argument("--out", default="/mnt/hgfs/mvs_shots/armor/Video_live.avi")
    ap.add_argument("--snapshot", default="", help="额外保存一帧 jpg")
    ap.add_argument("--exposure", type=float, default=20000.0, help="曝光时间 us")
    ap.add_argument("--gain", type=float, default=8.0, help="增益 dB")
    args = ap.parse_args()

    ret = MvCamera.MV_CC_Initialize()                     # noqa: F405
    print("SDK init:", hex(ret & 0xffffffff))
    dev_list = MV_CC_DEVICE_INFO_LIST()                   # noqa: F405
    tlayer = (MV_GIGE_DEVICE | MV_USB_DEVICE |            # noqa: F405
              MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE |  # noqa: F405
              MV_GENTL_XOF_DEVICE | MV_GENTL_XOC_DEVICE)  # noqa: F405
    ret = MvCamera.MV_CC_EnumDevices(tlayer, dev_list)    # noqa: F405
    print("enum devices:", hex(ret & 0xffffffff), dev_list.nDeviceNum)
    if ret != 0 or dev_list.nDeviceNum == 0:
        raise SystemExit("未枚举到相机，请检查 USB 直通与 MVS 客户端占用情况")

    info = cast(dev_list.pDeviceInfo[0],                  # noqa: F405
                POINTER(MV_CC_DEVICE_INFO)).contents      # noqa: F405
    cam = MvCamera()                                      # noqa: F405
    if cam.MV_CC_CreateHandle(info) != 0:
        raise SystemExit("创建相机句柄失败")
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)    # noqa: F405
    if ret != 0:
        print("打开设备失败:", hex(ret & 0xffffffff),
              "（0x80000203 表示 MVS 客户端正独占相机，请先关闭它）")
        raise SystemExit(2)

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)  # noqa: F405
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)  # noqa: F405
    cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    cam.MV_CC_SetFloatValue("ExposureTime", args.exposure)
    cam.MV_CC_SetEnumValue("GainAuto", 0)
    cam.MV_CC_SetFloatValue("Gain", args.gain)
    if cam.MV_CC_StartGrabbing() != 0:
        raise SystemExit("开始取流失败")

    st = MV_FRAME_OUT()                                   # noqa: F405
    memset(byref(st), 0, sizeof(st))                      # noqa: F405
    writer = None
    count = 0
    try:
        for i in range(args.frames):
            ret = cam.MV_CC_GetImageBuffer(st, 1000)
            if ret != 0 or not st.pBufAddr:
                continue
            fi = st.stFrameInfo
            w, h = fi.nWidth, fi.nHeight
            buf = string_at(st.pBufAddr, fi.nFrameLen)     # noqa: F405
            frame = to_bgr(buf, w, h, fi.enPixelType)
            if writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                            exist_ok=True)
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"MJPG"),
                    args.fps, (w, h))
                print("capture:", w, "x", h, "->", args.out)
            writer.write(frame)
            if args.snapshot and i == args.frames // 2:
                cv2.imwrite(args.snapshot, frame)
            count += 1
            cam.MV_CC_FreeImageBuffer(st)
    finally:
        if writer is not None:
            writer.release()
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()                         # noqa: F405
    print("DONE: %d frames -> %s" % (count, args.out))


if __name__ == "__main__":
    main()
