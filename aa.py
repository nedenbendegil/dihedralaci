import numpy as np

EPSILON = 1e-9

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        return v
    return v / norm

def are_parallel(v1, v2, tolerance=EPSILON):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < tolerance or v2_norm < tolerance:
        return v1_norm < tolerance and v2_norm < tolerance
        
    cross_prod_norm = np.linalg.norm(np.cross(normalize(v1), normalize(v2)))
    return np.isclose(cross_prod_norm, 0.0, atol=tolerance)

def are_orthogonal(v1, v2, tolerance=EPSILON):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < tolerance or v2_norm < tolerance:
         return True # zero vector hep orthogonal 
         
    # vektorler 0 mi
    dot_prod_normalized = np.dot(normalize(v1), normalize(v2))
    return np.isclose(dot_prod_normalized, 0.0, atol=tolerance)

n1 = np.array([0., 0., 1.])
n2 = normalize(np.array([1., -1., 0.]))


print(f"Plane 1 Normal (n1): {n1}")
print(f"Plane 2 Normal (n2): {n2}")
dir_L = np.cross(n1, n2)
if np.linalg.norm(dir_L) < EPSILON:
    print("\nbruh.")
    exit()
dir_L = normalize(dir_L)
print(f"dir (dir_L): {dir_L}")

v1 = np.cross(n1, dir_L) 
if np.linalg.norm(v1) < EPSILON:
     print("v1 calculation non zero.")
v1 = normalize(v1)
print(f"ray 1 dirction (v1 - in P1, perp to L): {v1}")

v2 = np.cross(n2, dir_L) # ok
if np.linalg.norm(v2) < EPSILON:
     print("Warning: v2 calculation resulted in near-zero vector. Check inputs/geometry.")
v2 = normalize(v2)
print(f"Ray 2 Direction (v2 - in P2, perp to L): {v2}")

n_Q = np.cross(v1, v2)
if np.linalg.norm(n_Q) < EPSILON:
    print("\nv1 v2 parrl angle 0 or 180 de).")
    n_Q = dir_L # ? dogru mu
else:
    n_Q = normalize(n_Q)
print(f"Dihedral Plane Normal (n_Q): {n_Q}")

print("\n---  ---")

print("Q spanned by v1, v2")

print(f"   Sanity Check: v1 orthogonal to n1? {are_orthogonal(v1, n1)}")
print(f"   Sanity Check: v1 orthogonal to dir_L? {are_orthogonal(v1, dir_L)}")
print(f"   Sanity Check: v2 orthogonal to n2? {are_orthogonal(v2, n2)}")
print(f"   Sanity Check: v2 orthogonal to dir_L? {are_orthogonal(v2, dir_L)}")

is_Q_perp_L = are_parallel(n_Q, dir_L)
print(f"2. Is Plane Q perpendicular to Line L? {is_Q_perp_L}")
if not is_Q_perp_L:
     dot_abs = abs(np.dot(normalize(n_Q), normalize(dir_L)))
     print(f"   (Debug: |dot(n_Q_norm, dir_L_norm)| = {dot_abs:.4f} - should be close to 1 if parallel)")


angle_rad = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) # aci v1, v2
is_degenerate_angle = np.isclose(angle_rad, 0.0, atol=EPSILON) or \
                      np.isclose(angle_rad, np.pi, atol=EPSILON)

is_independent = True
if is_degenerate_angle:
    print("3. Is Plane Q independent? No (Dihedral angle is 0 or 180 degrees)")
    is_independent = False
else:
    if are_parallel(n_Q, n1) or are_parallel(n_Q, n2):
         print("3. a 1 dogru cikti")
    else:
         print("3. a 2 yanlis hoca")

if not is_degenerate_angle:
    angle_deg = np.degrees(angle_rad)
    print(f"\nDihedral Angle calculated: {angle_deg:.2f} derece")
else:
    print(f"\nDihedral Angle calculated: {np.degrees(angle_rad):.2f} ddc (0 or 180)")
