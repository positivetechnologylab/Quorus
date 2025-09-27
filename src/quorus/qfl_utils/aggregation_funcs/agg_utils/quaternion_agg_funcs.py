"""### Quaternion mean"""

from scipy.spatial.transform import Rotation as R

def euler_to_quat(phi, theta, omega):
    """
    Converts Euler angles (phi, theta, omega) corresponding to
    qml.Rot(phi, theta, omega) = R_z(omega) R_y(theta) R_z(phi)
    to a quaternion.

    We use the 'zyz' convention in scipy so that:
       quat = R.from_euler('zyz', [omega, theta, phi]).as_quat()
    Note: as_quat() returns the quaternion in [x, y, z, w] order.
    """
    return R.from_euler('zyz', [omega, theta, phi]).as_quat()

def quat_to_euler(q):
    """
    Converts a quaternion back to Euler angles using the 'zyz' convention.
    The scipy routine returns angles in the order [omega, theta, phi];
    we then rearrange them to (phi, theta, omega) for use with qml.Rot.
    """
    euler = R.from_quat(q).as_euler('zyz')
    return euler[2], euler[1], euler[0]