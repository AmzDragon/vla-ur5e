import numpy as np


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero-length vector")
    return v / n


def look_at_xyaxes(camera_pos, target_pos, up=(0, 0, 1)):
    """
    输入:
        camera_pos: 相机位置, 例如 [1, 0, 1]
        target_pos: 相机看向的位置, 例如 [0, 0, 0]
        up: 世界坐标系的上方向, MuJoCo 通常用 z-up, 即 [0, 0, 1]

    输出:
        xyaxes: 长度为 6 的 list，可直接写入 MuJoCo camera 的 xyaxes
                [x_axis(3), y_axis(3)]
    """

    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # MuJoCo/OpenGL 相机看向 -z 方向
    z_axis = normalize(camera_pos - target_pos)

    # 如果视线方向和 up 几乎平行，换一个 up，避免叉乘接近 0
    if abs(np.dot(z_axis, normalize(up))) > 0.999:
        up = np.array([0, 1, 0], dtype=np.float64)

    # 相机 x 轴：图像右方
    x_axis = normalize(np.cross(up, z_axis))

    # 相机 y 轴：图像上方
    y_axis = normalize(np.cross(z_axis, x_axis))

    xyaxes = np.concatenate([x_axis, y_axis])
    return xyaxes.tolist()


def look_at_xyaxes_str(camera_pos, target_pos, up=(0, 0, 1), precision=6):
    xyaxes = look_at_xyaxes(camera_pos, target_pos, up)
    return " ".join(f"{v:.{precision}f}" for v in xyaxes)

pos = [0.6 ,-0.5 ,1.3]
target = [0.0, 0.0, 0.5] # 0.7 桌子

xyaxes = look_at_xyaxes_str(pos, target)
print(xyaxes)