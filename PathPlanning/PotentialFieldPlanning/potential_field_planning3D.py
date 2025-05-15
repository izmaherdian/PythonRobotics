import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameter global
KP = 5.0      # attractive potential gain
ETA = 100.0   # repulsive potential gain
AREA_WIDTH = 10.0  # lebar area [m]
OSC_LEN = 10        # panjang window deteksi osilasi (ubah jadi integer)
show_animation = True

def calc_potential_field_3d_spheres(
    gx, gy, gz,
    spheres,    # list of tuples (ox, oy, oz, r_obs)
    reso, rr,
    sx, sy, sz
):
    # kumpulkan array pusat-pusat bola
    centers = np.array([[o, p, q] for (o, p, q, _) in spheres])
    
    # bounding box sama seperti dulu, tapi pakai centers
    minx = min(centers[:,0].min(), sx, gx) - AREA_WIDTH/2
    miny = min(centers[:,1].min(), sy, gy) - AREA_WIDTH/2
    minz = min(centers[:,2].min(), sz, gz) - AREA_WIDTH/2
    maxx = max(centers[:,0].max(), sx, gx) + AREA_WIDTH/2
    maxy = max(centers[:,1].max(), sy, gy) + AREA_WIDTH/2
    maxz = max(centers[:,2].max(), sz, gz) + AREA_WIDTH/2

    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))
    zw = int(round((maxz - minz) / reso))

    pmap = np.zeros((xw, yw, zw))

    for ix in range(xw):
        x = minx + ix * reso
        for iy in range(yw):
            y = miny + iy * reso
            for iz in range(zw):
                z = minz + iz * reso

                # Atraktif
                dist_goal = np.linalg.norm([x-gx, y-gy, z-gz])
                U_att = 0.5 * KP * dist_goal

                # Repulsif: cari jarak permukaan bola terdekat
                dmin = np.inf
                for ox, oy, oz, r_obs in spheres:
                    d_center = np.linalg.norm([x-ox, y-oy, z-oz])
                    # jika di dalam bola, d_surface = 0
                    d_surface = max(d_center - r_obs, 0.0)
                    dmin = min(dmin, d_surface)

                U_rep = 0.0
                if dmin <= rr:
                    dq = max(dmin, 1e-3)
                    U_rep = 0.5 * ETA * (1.0/dq - 1.0/rr)**2

                pmap[ix, iy, iz] = U_att + U_rep

    return pmap, (minx, miny, minz)


def get_motion_model_3d():
    return [
        (dx, dy, dz)
        for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
        if not (dx==0 and dy==0 and dz==0)
    ]


def detect_oscillation(prev, ix, iy, iz):
    prev.append((ix, iy, iz))
    if len(prev) > OSC_LEN:
        prev.popleft()
    return len(prev) != len(set(prev))


def potential_field_planning_3d(
    sx, sy, sz,
    gx, gy, gz,
    spheres,    # daftar (ox,oy,oz,radius)
    reso, rr
):
    # hitung peta potensi
    pmap, (minx, miny, minz) = calc_potential_field_3d_spheres(
        gx, gy, gz, spheres, reso, rr, sx, sy, sz
    )

    # indeks voxel start
    ix = int(round((sx - minx) / reso))
    iy = int(round((sy - miny) / reso))
    iz = int(round((sz - minz) / reso))
    path = [(sx, sy, sz)]

    # inisialisasi visualisasi
    if show_animation:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot bola obstacle (titik pusat saja)
        for ox, oy, oz, r in spheres:
            ax.scatter([ox], [oy], [oz], c='k', s=(r*100)**2, alpha=0.5)
        ax.scatter([sx], [sy], [sz], c='b', marker='^', s=100, label='Start')
        ax.scatter([gx], [gy], [gz], c='r', marker='*', s=100, label='Goal')
        ax.legend()
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    motion = get_motion_model_3d()
    prev = deque()

    while True:
        # pilih tetangga potensi terendah
        min_p, next_idx = float('inf'), None
        for dx, dy, dz in motion:
            nx, ny, nz = ix+dx, iy+dy, iz+dz
            if 0 <= nx < pmap.shape[0] and 0 <= ny < pmap.shape[1] and 0 <= nz < pmap.shape[2]:
                pot = pmap[nx, ny, nz]
            else:
                pot = float('inf')
            if pot < min_p:
                min_p, next_idx = pot, (nx, ny, nz)

        if next_idx is None:
            print("Terjebak atau keluar area!")
            break

        ix, iy, iz = next_idx
        x = minx + ix * reso
        y = miny + iy * reso
        z = minz + iz * reso
        path.append((x, y, z))

        # cek osilasi
        if detect_oscillation(prev, ix, iy, iz):
            print(f"Osilasi terdeteksi pada voxel {(ix,iy,iz)}")
            break

        # plot titik baru
        if show_animation:
            ax.scatter([x], [y], [z], c='g', marker='o', s=20)
            plt.draw(); plt.pause(0.01)

        # cek goal
        if np.linalg.norm([x-gx, y-gy, z-gz]) < reso:
            print("Goal tercapai!")
            break

    if show_animation:
        plt.show()

    return path


# Contoh penggunaan
if __name__ == "__main__":
    sx, sy, sz = 0.0, 0.0, 0.0
    gx, gy, gz = 10.0, 8.0, 6.0
    # daftar bola obstacle: (x, y, z, radius)
    spheres = [
        (5.0, 5.0, 5.0, 0.5),
        (10.0, 6.0, 4.0, 0.2),
        (5.0, 6.0, 3.0, 0.1),
    ]
    reso = 0.7
    rr = 5.0

    path3d = potential_field_planning_3d(
        sx, sy, sz,
        gx, gy, gz,
        spheres,
        reso, rr
    )
    print("Path 3D:", path3d)
