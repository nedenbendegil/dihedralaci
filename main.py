# fuck love
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display, clear_output
import sys
import traceback

#(Bunlari Ayni Tut!!!)
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def get_plane_points(origin, normal, basis1, basis2, size=2):
    u_vals = np.linspace(-size/2, size/2, 10)
    v_vals = np.linspace(-size/2, size/2, 10)
    U, V = np.meshgrid(u_vals, v_vals)
    X = origin[0] + U * basis1[0] + V * basis2[0]
    Y = origin[1] + U * basis1[1] + V * basis2[1]
    Z = origin[2] + U * basis1[2] + V * basis2[2]
    return X, Y, Z

# SADECE BIR KEZ, olustur update disinda
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
# bos plot gosterme amk oglu
plt.close(fig)

# buradayken soylemek isterim ki,
# seni cok seviyordum,
# beni bos bir kuyuya surukeldin,
# ve dustugum icin lanetler okudun

output_area = widgets.Output()

# duzelttim ok sorun yok
def update_plot(alpha_deg, beta_deg, point_dist, elev, azim):
    global fig, ax 

    with output_area:
        clear_output(wait=True) # uhh
        ax.cla() # asli tam bi pislik axxxxx

        try:
            n1 = np.array([0.0, 0.0, 1.0])
            origin = np.array([0.0, 0.0, 0.0])
            p1_basis1 = np.array([1.0, 0.0, 0.0])
            p1_basis2 = np.array([0.0, 1.0, 0.0])

            alpha = np.radians(alpha_deg)
            beta = np.radians(beta_deg)
            beta_for_basis = beta + 0.001 if np.isclose(beta, 0) or np.isclose(beta, np.pi) else beta

            nx = np.sin(beta) * np.cos(alpha)
            ny = np.sin(beta) * np.sin(alpha)
            nz = np.cos(beta)
            n2 = normalize(np.array([nx, ny, nz]))

            dot_prod = np.dot(n1, n2)
            if np.allclose(np.abs(dot_prod), 1.0, atol=1e-6):
                ax.text(0, 0, 0.5, "Planes are parallel or identical", color='red',
                        ha='center', va='center', fontsize=12, zorder=10)
                plane_size_viz = 3
                X1_p, Y1_p, Z1_p = get_plane_points(origin, n1, p1_basis1, p1_basis2, size=plane_size_viz)
                ax.plot_surface(X1_p, Y1_p, Z1_p, alpha=0.3, color='blue', rstride=5, cstride=5)
                lim_p = plane_size_viz * 0.6
                ax.set_xlim([-lim_p, lim_p]); ax.set_ylim([-lim_p, lim_p]); ax.set_zlim([-lim_p, lim_p])
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
                ax.set_title("paralel")
                # Set view angle even for parallel case
                ax.view_init(elev=elev, azim=azim)
                ax.set_aspect('equal', adjustable='box')
                display(fig)
                return # Stop

            nx_b = np.sin(beta_for_basis) * np.cos(alpha)
            ny_b = np.sin(beta_for_basis) * np.sin(alpha)
            nz_b = np.cos(beta_for_basis)
            n2_for_basis = normalize(np.array([nx_b, ny_b, nz_b])) # Allaha havala ediyorum kizim seni
            dummy = np.array([1.0, 0.0, 0.0])
            if np.allclose(np.abs(np.dot(n2_for_basis, dummy)), 1.0):
                dummy = np.array([0.0, 1.0, 0.0])
            p2_basis1 = normalize(np.cross(n2_for_basis, dummy))
            p2_basis2 = normalize(np.cross(n2_for_basis, p2_basis1))

            L = normalize(np.cross(n1, n2)) # offf
            P = origin + point_dist * L
            v1 = normalize(np.cross(L, n1))
            v2 = normalize(np.cross(L, n2)) # dikey olarak al planeleri

            cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
            dihedral_angle_rad = np.arccos(cos_theta)
            dihedral_angle_deg = np.degrees(dihedral_angle_rad)

            n3 = normalize(np.cross(v1, v2))
            m_basis1 = v1
            m_basis2 = v2

            plane_size = 3
            ray_length = max(0.5, 0.6 * point_dist)
            measure_plane_size = ray_length * 1.2
            X1, Y1, Z1 = get_plane_points(origin, n1, p1_basis1, p1_basis2, size=plane_size)
            X2, Y2, Z2 = get_plane_points(origin, n2, p2_basis1, p2_basis2, size=plane_size)
            line_pts = np.array([origin - L * plane_size * 0.7, origin + L * plane_size * 0.7])
            X3, Y3, Z3 = get_plane_points(P, n3, m_basis1, m_basis2, size=measure_plane_size)

            ax.plot_surface(X1, Y1, Z1, alpha=0.3, color='blue', rstride=5, cstride=5, label='Plane 1 (z=0)')
            ax.plot_surface(X2, Y2, Z2, alpha=0.3, color='red', rstride=5, cstride=5, label='Plane 2 (Adjustable)')
            ax.plot(line_pts[:,0], line_pts[:,1], line_pts[:,2], 'k--', lw=2, label='Line of Intersection')
            ax.scatter(P[0], P[1], P[2], color='black', s=60, label=f'Point P (Dist={point_dist:.1f})', zorder=5)
            ax.quiver(P[0], P[1], P[2], v1[0], v1[1], v1[2], length=ray_length, color='cyan', lw=3, label='Ray 1 (in Plane 1)', zorder=6)
            ax.quiver(P[0], P[1], P[2], v2[0], v2[1], v2[2], length=ray_length, color='magenta', lw=3, label='Ray 2 (in Plane 2)', zorder=6)
            ax.plot_surface(X3, Y3, Z3, alpha=0.4, color='green', rstride=1, cstride=1, label='Measurement Plane', zorder=4)

            lim = plane_size * 0.6
            ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
            ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
            ax.set_title(f'Dihedral Angle Visualization (Angle = {dihedral_angle_deg:.1f}°)', fontsize=14)

            # sldier
            ax.view_init(elev=elev, azim=azim)

            ax.set_aspect('equal', adjustable='box') # aspect ratio degismemeli

            display(fig)

        except Exception as e:
             # HASSSSIKTIR LAN
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
            print(f"HATA GELDI : update_plot: {e}", file=sys.stderr)
            print("Traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
            try:
                ax.view_init(elev=elev, azim=azim)
                display(fig)
            except:
                 print("heheheheheheheheheheheehehe.", file=sys.stderr)


# --- DONDUR ---
alpha_slider = widgets.FloatSlider(min=0, max=360, step=1, value=30, description='Plane 2 Rot (α°):', continuous_update=False, layout=widgets.Layout(width='500px'))
beta_slider = widgets.FloatSlider(min=1, max=179, step=1, value=55, description='Plane 2 Tilt (β°):', continuous_update=False, layout=widgets.Layout(width='500px'))
point_dist_slider = widgets.FloatSlider(min=0.0, max=1.5, step=0.1, value=0.8, description='Point P Dist:', continuous_update=False, layout=widgets.Layout(width='500px'))

elev_slider = widgets.FloatSlider(min=-90, max=90, step=5, value=20, description='Camera Elev (°):', continuous_update=False, layout=widgets.Layout(width='500px'))
azim_slider = widgets.FloatSlider(min=0, max=360, step=5, value=-60, description='Camera Azim (°):', continuous_update=False, layout=widgets.Layout(width='500px'))

ui = widgets.VBox([
    widgets.Label("Plane Controls:"),
    alpha_slider, beta_slider, point_dist_slider,
    widgets.HTML("<hr>"), # TEEEEEST
    widgets.Label("Camera Controls:"),
    elev_slider, azim_slider
])

# elev ve azim yezidi isimleri gibi aq
interactive_plot = widgets.interactive_output(update_plot, {
    'alpha_deg': alpha_slider,
    'beta_deg': beta_slider,
    'point_dist': point_dist_slider,
    'elev': elev_slider,
    'azim': azim_slider
})

display(ui, output_area)

update_plot(alpha_slider.value, beta_slider.value, point_dist_slider.value, elev_slider.value, azim_slider.value)
