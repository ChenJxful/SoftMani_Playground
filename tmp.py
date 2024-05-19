import numpy as np

def calculate_rotation_quaternion_from_vectors(ref, target):
    """
    Calculate the rotation quaternion that aligns the reference vector 'ref' to the target vector 'target'.
    The rotation is such that the angle between 'ref' and 'target' is minimized and does not exceed 90 degrees.
    
    Args:
    ref (list or array): Reference vector.
    target (list or array): Target vector to which the reference vector should be aligned.
    
    Returns:
    numpy.array: Quaternion [x, y, z, w] representing the rotation.
    """
    ref = np.array(ref)
    target = np.array(target)
    
    # Normalize the vectors
    ref = ref / np.linalg.norm(ref)
    target = target / np.linalg.norm(target)
    
    # Calculate the cross product to find the rotation axis
    axis = np.cross(ref, target)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm == 0:
        # Vectors are parallel, no rotation needed 
        return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion

    # Normalize the rotation axis
    axis = axis / axis_norm
    
    # Calculate the cosine of the angle using the dot product
    cos_theta = np.dot(ref, target)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure within valid range
    theta = np.arccos(cos_theta)
    
    # Adjust angle to not exceed 90 degrees
    if theta > np.pi / 2:
        theta = np.pi - theta  # Reflect the angle over the 90 degrees boundary
        axis = -axis           # Reverse the rotation direction
    
    # Calculate quaternion using half-angle trigonometric identities
    w = np.cos(theta / 2)
    sin_half_theta = np.sin(theta / 2)
    x = axis[0] * sin_half_theta
    y = axis[1] * sin_half_theta
    z = axis[2] * sin_half_theta
    
    return np.array([x, y, z, w])

# Example vectors
ref_vector = [1, 0, 0]
target_vector = [1, 1, 0]
target_vector2 = [-1, -1, 0]
# Calculate quaternion
quat = calculate_rotation_quaternion_from_vectors(ref_vector, target_vector)
print("Quaternion:", quat)

quat2 = calculate_rotation_quaternion_from_vectors(ref_vector, target_vector2)
print("Quaternion2:", quat2)