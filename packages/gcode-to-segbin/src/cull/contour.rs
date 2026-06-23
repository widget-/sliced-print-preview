use std::collections::HashMap;
use crate::segment::Segment;
use crate::stl::STLMesh;

// ── Contour types ──

struct Contour {
    verts: Vec<[f32; 2]>,
}

struct ContourGrid {
    #[allow(dead_code)]
    cell_size: f32,
    inv_cell: f32,
    bbox_min: [f32; 2],
    dims: [usize; 2],
    cells: Vec<Vec<(usize, usize, usize)>>,
    all_verts: Vec<[f32; 2]>,
    vert_offsets: Vec<usize>,
}

impl ContourGrid {
    fn point_inside(&self, p: [f32; 2]) -> bool {
        let mut winding = 0i32;
        for ci in 0..self.vert_offsets.len() {
            let off = self.vert_offsets[ci];
            let len = if ci + 1 < self.vert_offsets.len() { self.vert_offsets[ci + 1] - off } else { self.all_verts.len() - off };
            if len < 3 { continue; }
            for i in 0..len {
                let a = self.all_verts[off + i];
                let b = self.all_verts[off + (i + 1) % len];
                if (a[1] > p[1]) != (b[1] > p[1]) {
                    let x_cross = a[0] + (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1] + 1e-12);
                    if x_cross > p[0] { winding += 1; }
                }
            }
        }
        winding % 2 != 0
    }

    fn from_contours(contours: &[Contour], cell_size: f32) -> Self {
        let mut bmin = [f32::MAX; 2];
        let mut bmax = [f32::MIN; 2];
        for c in contours {
            for v in &c.verts {
                bmin[0] = bmin[0].min(v[0]); bmin[1] = bmin[1].min(v[1]);
                bmax[0] = bmax[0].max(v[0]); bmax[1] = bmax[1].max(v[1]);
            }
        }
        bmin[0] -= cell_size * 0.1; bmin[1] -= cell_size * 0.1;
        bmax[0] += cell_size * 0.1; bmax[1] += cell_size * 0.1;
        let inv_cell = 1.0 / cell_size;
        let dims = [
            ((bmax[0] - bmin[0]) * inv_cell).ceil() as usize + 1,
            ((bmax[1] - bmin[1]) * inv_cell).ceil() as usize + 1,
        ];
        let cell_count = dims[0] * dims[1];
        let mut cells: Vec<Vec<(usize, usize, usize)>> = (0..cell_count).map(|_| Vec::new()).collect();

        let mut all_verts = Vec::new();
        let mut vert_offsets = Vec::new();
        for (ci, c) in contours.iter().enumerate() {
            vert_offsets.push(all_verts.len());
            for v in &c.verts { all_verts.push(*v); }
            let off = vert_offsets[ci];
            for ei in 0..c.verts.len() {
                let a = c.verts[ei];
                let b = c.verts[(ei + 1) % c.verts.len()];
                let mn = [a[0].min(b[0]), a[1].min(b[1])];
                let mx = [a[0].max(b[0]), a[1].max(b[1])];
                let cx = ((mn[0] - bmin[0]) * inv_cell).floor() as isize;
                let cy = ((mn[1] - bmin[1]) * inv_cell).floor() as isize;
                let ex = ((mx[0] - bmin[0]) * inv_cell).ceil() as isize;
                let ey = ((mx[1] - bmin[1]) * inv_cell).ceil() as isize;
                for gy in cy..=ey {
                    for gx in cx..=ex {
                        if gx < 0 || gy < 0 || gx as usize >= dims[0] || gy as usize >= dims[1] { continue; }
                        let key = gy as usize * dims[0] + gx as usize;
                        cells[key].push((ci, off + ei, off + ((ei + 1) % c.verts.len())));
                    }
                }
            }
        }

        ContourGrid { cell_size, inv_cell, bbox_min: bmin, dims, cells, all_verts, vert_offsets }
    }

    fn nearest_edge_distance(&self, p: [f32; 2]) -> (f32, bool) {
        let cx = ((p[0] - self.bbox_min[0]) * self.inv_cell).floor() as isize;
        let cy = ((p[1] - self.bbox_min[1]) * self.inv_cell).floor() as isize;
        let rad = 5isize;
        let mut best_sq = 1e30f32;
        let mut winding = 0i32;

        for dy in -rad..=rad {
            for dx in -rad..=rad {
                let gx = cx + dx;
                let gy = cy + dy;
                if gx < 0 || gy < 0 || gx as usize >= self.dims[0] || gy as usize >= self.dims[1] { continue; }
                let key = gy as usize * self.dims[0] + gx as usize;
                for &(_, ea, eb) in &self.cells[key] {
                    let a = self.all_verts[ea];
                    let b = self.all_verts[eb];
                    let ab = [b[0] - a[0], b[1] - a[1]];
                    let ap = [p[0] - a[0], p[1] - a[1]];
                    let t = (ap[0] * ab[0] + ap[1] * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1] + 1e-12);
                    let t = t.clamp(0.0, 1.0);
                    let qx = a[0] + t * ab[0];
                    let qy = a[1] + t * ab[1];
                    let dx_ = p[0] - qx;
                    let dy_ = p[1] - qy;
                    let d2 = dx_ * dx_ + dy_ * dy_;
                    if d2 < best_sq { best_sq = d2; }
                    if (a[1] > p[1]) != (b[1] > p[1]) {
                        let x_cross = a[0] + (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1] + 1e-12);
                        if x_cross > p[0] { winding += 1; }
                    }
                }
            }
        }
        (best_sq, winding % 2 != 0)
    }
}

// ── Height map ──

struct HeightMap {
    origin: [f32; 2],
    cell_size: f32,
    dims: [usize; 2],
    z_max: Vec<f32>,
    z_min: Vec<f32>,
}

fn build_height_map(mesh: &STLMesh, cell_size: f32) -> HeightMap {
    let mut bmin = [f32::MAX; 2];
    let mut bmax = [f32::MIN; 2];
    for tri in &mesh.triangles {
        for v in tri {
            bmin[0] = bmin[0].min(v[0]); bmin[1] = bmin[1].min(v[1]);
            bmax[0] = bmax[0].max(v[0]); bmax[1] = bmax[1].max(v[1]);
        }
    }
    bmin[0] -= cell_size; bmin[1] -= cell_size;
    bmax[0] += cell_size; bmax[1] += cell_size;
    let dims = [
        ((bmax[0] - bmin[0]) / cell_size).ceil() as usize,
        ((bmax[1] - bmin[1]) / cell_size).ceil() as usize,
    ];
    let mut z_max = vec![0.0f32; dims[0] * dims[1]];
    let mut z_min = vec![1e30f32; dims[0] * dims[1]];

    for tri in &mesh.triangles {
        let mut tmin = [f32::MAX; 2]; let mut tmax = [f32::MIN; 2];
        for v in tri {
            tmin[0] = tmin[0].min(v[0]); tmin[1] = tmin[1].min(v[1]);
            tmax[0] = tmax[0].max(v[0]); tmax[1] = tmax[1].max(v[1]);
        }
        let gx0 = ((tmin[0] - bmin[0]) / cell_size).floor() as isize;
        let gy0 = ((tmin[1] - bmin[1]) / cell_size).floor() as isize;
        let gx1 = ((tmax[0] - bmin[0]) / cell_size).ceil() as isize;
        let gy1 = ((tmax[1] - bmin[1]) / cell_size).ceil() as isize;
        for gy in gy0..=gy1 {
            for gx in gx0..=gx1 {
                if gx < 0 || gy < 0 || gx as usize >= dims[0] || gy as usize >= dims[1] { continue; }
                let cx = bmin[0] + (gx as f32 + 0.5) * cell_size;
                let cy = bmin[1] + (gy as f32 + 0.5) * cell_size;
                let v0 = tri[0]; let v1 = tri[1]; let v2 = tri[2];
                let d00 = (v1[0]-v0[0])*(v1[0]-v0[0]) + (v1[1]-v0[1])*(v1[1]-v0[1]);
                let d01 = (v1[0]-v0[0])*(v2[0]-v0[0]) + (v1[1]-v0[1])*(v2[1]-v0[1]);
                let d11 = (v2[0]-v0[0])*(v2[0]-v0[0]) + (v2[1]-v0[1])*(v2[1]-v0[1]);
                let d20 = (cx-v0[0])*(v1[0]-v0[0]) + (cy-v0[1])*(v1[1]-v0[1]);
                let d21 = (cx-v0[0])*(v2[0]-v0[0]) + (cy-v0[1])*(v2[1]-v0[1]);
                let denom = d00 * d11 - d01 * d01;
                if denom.abs() < 1e-12 { continue; }
                let v = (d11 * d20 - d01 * d21) / denom;
                let w = (d00 * d21 - d01 * d20) / denom;
                let u = 1.0 - v - w;
                if u >= 0.0 && v >= 0.0 && w >= 0.0 {
                    let z = u * v0[2] + v * v1[2] + w * v2[2];
                    let idx = gy as usize * dims[0] + gx as usize;
                    if z > z_max[idx] { z_max[idx] = z; }
                    if z < z_min[idx] { z_min[idx] = z; }
                }
            }
        }
    }
    for v in z_min.iter_mut() { if *v > 1e29 { *v = 0.0; } }
    HeightMap { origin: bmin, cell_size, dims, z_max, z_min }
}

// ── STL slicing ──

fn slice_stl_at_z(mesh: &STLMesh, z: f32) -> Vec<[[f32; 2]; 2]> {
    let eps = 1e-8;
    let mut segs = Vec::new();
    for tri in &mesh.triangles {
        let mut hits = Vec::new();
        for i in 0..3 {
            let j = (i + 1) % 3;
            let a = tri[i][2];
            let b = tri[j][2];
            if (a - z).abs() < eps {
                if (b - z).abs() >= eps {
                    hits.push([tri[i][0], tri[i][1]]);
                }
            } else if (b - z).abs() < eps {
                if (a - z).abs() >= eps {
                    hits.push([tri[j][0], tri[j][1]]);
                }
            } else if (a < z && b > z) || (a > z && b < z) {
                let t = (z - a) / (b - a);
                hits.push([tri[i][0] + t * (tri[j][0] - tri[i][0]),
                           tri[i][1] + t * (tri[j][1] - tri[i][1])]);
            }
        }
        if hits.len() >= 2 {
            segs.push([hits[0], hits[1]]);
        }
    }
    segs
}

fn connect_contours(segs: &[[[f32; 2]; 2]]) -> Vec<Contour> {
    if segs.is_empty() { return Vec::new(); }
    let eps = 0.01;
    let key = |x: f32, y: f32| -> u64 {
        let ix = (x / eps).round() as i64;
        let iy = (y / eps).round() as i64;
        (ix as u64) << 32 | (iy as u64)
    };
    let mut adj: HashMap<u64, Vec<(usize, bool)>> = HashMap::new();
    for (i, seg) in segs.iter().enumerate() {
        let ka = key(seg[0][0], seg[0][1]);
        let kb = key(seg[1][0], seg[1][1]);
        adj.entry(ka).or_default().push((i, true));
        adj.entry(kb).or_default().push((i, false));
    }

    let mut used = vec![false; segs.len()];
    let mut contours: Vec<Contour> = Vec::new();
    for start_i in 0..segs.len() {
        if used[start_i] { continue; }
        let mut verts: Vec<[f32; 2]> = Vec::new();
        let mut seg_i = start_i;
        let mut forward = true;
        loop {
            used[seg_i] = true;
            let seg = &segs[seg_i];
            if forward { verts.push(seg[0]); } else { verts.push(seg[1]); }
            let tip = if forward { seg[1] } else { seg[0] };
            let tk = key(tip[0], tip[1]);
            let mut found = false;
            if let Some(neighbors) = adj.get(&tk) {
                for &(ni, nfwd) in neighbors {
                    if ni == seg_i || used[ni] { continue; }
                    seg_i = ni; forward = nfwd; found = true; break;
                }
            }
            if !found || seg_i == start_i { break; }
        }
        if verts.len() >= 3 { contours.push(Contour { verts }); }
    }
    contours
}

// ── Public API ──

pub fn cull_with_contours(
    segments: &mut [Segment], mesh: &STLMesh, max_width: f32, layer_h: f32,
    gc_center: [f32; 3], stl_center: [f32; 3], swap_xy: bool,
) {
    let t0 = std::time::Instant::now();
    let max_z = {
        let mut mz = 0.0f32;
        for tri in &mesh.triangles {
            for v in tri { if v[2] > mz { mz = v[2]; } }
        }
        mz
    };
    let n_layers = (max_z / layer_h).ceil() as usize + 1;
    let mut layer_data: Vec<(f32, ContourGrid, HeightMap)> = Vec::with_capacity(n_layers);
    for k in 0..n_layers {
        let z = k as f32 * layer_h;
        let segs = slice_stl_at_z(mesh, z);
        let contours = connect_contours(&segs);
        let grid = ContourGrid::from_contours(&contours, 3.5);
        let hmap = build_height_map(mesh, max_width * 0.5);
        layer_data.push((z, grid, hmap));
    }
    let slice_ms = t0.elapsed();
    eprintln!("  Sliced STL: {} layers in {}ms", n_layers, slice_ms.as_millis());

    let cull_start = std::time::Instant::now();
    let xy_thresh = max_width + 0.2;
    let z_thresh = (layer_h * 3.0 + 0.05).max(0.3);
    let mut culled = 0usize;
    let xform = |x: f32, y: f32, z: f32| -> [f32; 3] {
        if swap_xy {
            [stl_center[0] + (y - gc_center[1]),
             stl_center[1] - (x - gc_center[0]),
             z - (gc_center[2] - stl_center[2])]
        } else {
            [x - (gc_center[0] - stl_center[0]),
             y - (gc_center[1] - stl_center[1]),
             z - (gc_center[2] - stl_center[2])]
        }
    };
    for seg in segments.iter_mut() {
        let stl_s = xform(seg.data[0], seg.data[1], seg.data[2]);
        let stl_e = xform(seg.data[3], seg.data[4], seg.data[5]);
        let sx = stl_s[0]; let sy = stl_s[1]; let sz = stl_s[2];
        let ex = stl_e[0]; let ey = stl_e[1]; let ez = stl_e[2];
        let mz = (sz + ez) * 0.5;
        let layer_k = (mz / layer_h).round() as usize;
        if layer_k >= layer_data.len() { continue; }
        let (_, ref grid, ref hmap) = layer_data[layer_k];

        let step = xy_thresh.min(z_thresh);
        let dx = ex - sx; let dy = ey - sy; let dz = ez - sz;
        let seg_len = (dx * dx + dy * dy + dz * dz).sqrt();
        let n = (seg_len / step).ceil() as usize + 1;
        let mut keep = false;

        for i in 0..n {
            let t = 0.1 + (i as f32) * 0.8 / ((n - 1).max(1) as f32);
            let px = sx + t * dx;
            let py = sy + t * dy;
            let pz = sz + t * dz;

            let (d2, _inside) = grid.nearest_edge_distance([px, py]);
            if d2 < xy_thresh * xy_thresh { keep = true; break; }

            let hx = (px - hmap.origin[0]) / hmap.cell_size;
            let hy = (py - hmap.origin[1]) / hmap.cell_size;
            let ix = hx.floor() as isize;
            let iy = hy.floor() as isize;
            if ix >= 0 && iy >= 0 && (ix as usize) < hmap.dims[0] && (iy as usize) < hmap.dims[1] {
                let idx = iy as usize * hmap.dims[0] + ix as usize;
                let z_top = hmap.z_max[idx];
                let z_bot = hmap.z_min[idx];
                if (z_top > 0.0 && (pz - z_top).abs() < z_thresh)
                    || (z_bot > 0.0 && (pz - z_bot).abs() < z_thresh) {
                    keep = true; break;
                }
            }

            let max_scan = (layer_k + 1 + 20).min(layer_data.len());
            for uk in (layer_k + 1)..max_scan {
                if !layer_data[uk].1.point_inside([px, py]) {
                    keep = true; break;
                }
            }
            if keep { break; }
        }
        if !keep {
            seg.set_width(0.0);
            culled += 1;
        }
    }
    let query_ms = cull_start.elapsed();
    eprintln!("  Contour cull: {} / {} culled ({}ms)", culled, segments.len(), query_ms.as_millis());
}
