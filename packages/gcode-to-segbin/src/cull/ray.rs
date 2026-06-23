use bvh::aabb::{AABB, Bounded};
use bvh::bvh::{BVH, BVHNode};
use bvh::bounding_hierarchy::BHShape;
use bvh::ray::Ray;
use glam::Vec3;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::segment::{Role, Segment};
use crate::stl::STLMesh;

// ── Segment BVH ──

struct SegPrim {
    start: [f32;3],
    end: [f32;3],
    node_idx: usize,
}

impl Bounded for SegPrim {
    fn aabb(&self) -> AABB {
        let pad = 0.5;
        let bmin = Vec3::from([self.start[0].min(self.end[0])-pad, self.start[1].min(self.end[1])-pad, self.start[2].min(self.end[2])-pad]);
        let bmax = Vec3::from([self.start[0].max(self.end[0])+pad, self.start[1].max(self.end[1])+pad, self.start[2].max(self.end[2])+pad]);
        AABB::with_bounds(bmin, bmax)
    }
}

impl BHShape for SegPrim {
    fn set_bh_node_index(&mut self, idx: usize) { self.node_idx = idx; }
    fn bh_node_index(&self) -> usize { self.node_idx }
}

struct SegBVH {
    bvh: BVH,
    seg_start: Vec<[f32;3]>,
    seg_end: Vec<[f32;3]>,
}

impl SegBVH {
    fn from_segments(segments: &[Segment]) -> Self {
        let mut prims = Vec::with_capacity(segments.len());
        let mut seg_start = Vec::with_capacity(segments.len());
        let mut seg_end = Vec::with_capacity(segments.len());
        for seg in segments.iter() {
            let s = [seg.data[0], seg.data[1], seg.data[2]];
            let e = [seg.data[3], seg.data[4], seg.data[5]];
            seg_start.push(s); seg_end.push(e);
            prims.push(SegPrim { start: s, end: e, node_idx: 0 });
        }
        let bvh = BVH::build(&mut prims);
        // prims dropped here — not needed for traversal (use seg_start/seg_end by shape_index)
        SegBVH { bvh, seg_start, seg_end }
    }

    fn intersect(&self, ro: [f32;3], rd: [f32;3], hit_radius: f32) -> Option<(usize, f32)> {
        let ray = Ray::new(Vec3::from(ro), Vec3::from(rd));
        let nodes = &self.bvh.nodes;
        let mut best: Option<(usize, f32)> = None;
        let mut stack = [0usize; 64];
        let mut sp = 1usize;
        stack[0] = 0;
        while sp > 0 {
            sp -= 1;
            match nodes[stack[sp]] {
                BVHNode::Node { child_l_aabb, child_l_index, child_r_aabb, child_r_index, .. } => {
                    if ray.intersects_aabb(&child_l_aabb) {
                        stack[sp] = child_l_index;
                        sp += 1;
                    }
                    if ray.intersects_aabb(&child_r_aabb) {
                        stack[sp] = child_r_index;
                        sp += 1;
                    }
                }
                BVHNode::Leaf { shape_index, .. } => {
                    let (hit, t) = ray_seg_intersect(ro, rd, self.seg_start[shape_index], self.seg_end[shape_index], hit_radius);
                    if hit && t >= 0.0 && (best.is_none() || t < best.unwrap().1) {
                        best = Some((shape_index, t));
                    }
                }
            }
        }
        best
    }
}

// ── Ray-segment intersection ──

fn ray_seg_intersect(ro: [f32;3], rd: [f32;3], sa: [f32;3], sb: [f32;3], hit_radius: f32) -> (bool, f32) {
    let ab=[sb[0]-sa[0],sb[1]-sa[1],sb[2]-sa[2]]; let len_sq=ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2];
    if len_sq<1e-12{return(false,0.0);}
    let rd_dot_rd=rd[0]*rd[0]+rd[1]*rd[1]+rd[2]*rd[2];
    let rd_dot_ab=rd[0]*ab[0]+rd[1]*ab[1]+rd[2]*ab[2];
    let oc=[ro[0]-sa[0],ro[1]-sa[1],ro[2]-sa[2]];
    let rd_dot_oc=rd[0]*oc[0]+rd[1]*oc[1]+rd[2]*oc[2];
    let ab_dot_oc=ab[0]*oc[0]+ab[1]*oc[1]+ab[2]*oc[2];
    let denom=rd_dot_rd*len_sq-rd_dot_ab*rd_dot_ab;
    if denom.abs()<1e-12{return(false,0.0);}
    let t=(rd_dot_ab*ab_dot_oc-len_sq*rd_dot_oc)/denom;
    let u=(rd_dot_rd*ab_dot_oc-rd_dot_ab*rd_dot_oc)/denom;
    if t<0.0||u<0.0||u>1.0{return(false,0.0);}
    let pr=[ro[0]+t*rd[0],ro[1]+t*rd[1],ro[2]+t*rd[2]];
    let ps=[sa[0]+u*ab[0],sa[1]+u*ab[1],sa[2]+u*ab[2]];
    let d2=(pr[0]-ps[0]).powi(2)+(pr[1]-ps[1]).powi(2)+(pr[2]-ps[2]).powi(2);
    if d2>hit_radius*hit_radius{return(false,0.0);}
    (true,t)
}

// ── Triangle BVH for inside-volume test ──

struct TriP2 {
    idx: usize,
    v: [[f32;3];3],
    ni: usize,
}

impl Bounded for TriP2 {
    fn aabb(&self) -> AABB {
        let b=[self.v[0][0].min(self.v[1][0]).min(self.v[2][0])-0.01, self.v[0][1].min(self.v[1][1]).min(self.v[2][1])-0.01, self.v[0][2].min(self.v[1][2]).min(self.v[2][2])-0.01];
        let c=[self.v[0][0].max(self.v[1][0]).max(self.v[2][0])+0.01, self.v[0][1].max(self.v[1][1]).max(self.v[2][1])+0.01, self.v[0][2].max(self.v[1][2]).max(self.v[2][2])+0.01];
        AABB::with_bounds(Vec3::from(b), Vec3::from(c))
    }
}

impl BHShape for TriP2 {
    fn set_bh_node_index(&mut self, i:usize){self.ni=i;}
    fn bh_node_index(&self)->usize{self.ni}
}

// ── Triangle sampling for ray origins ──

fn tri_normal(tri: &[[f32;3];3]) -> [f32;3] {
    let ab = [tri[1][0]-tri[0][0], tri[1][1]-tri[0][1], tri[1][2]-tri[0][2]];
    let ac = [tri[2][0]-tri[0][0], tri[2][1]-tri[0][1], tri[2][2]-tri[0][2]];
    let n = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]];
    let l = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt();
    if l < 1e-10 { return [0.0, 0.0, 1.0]; }
    [n[0]/l, n[1]/l, n[2]/l]
}

fn sample_triangle_adaptive(
    tri: &[[f32;3];3], n: [f32;3],
    fine_spacing: f32, coarse_spacing: f32,
) -> Vec<([f32;3], [f32;3])> {
    let e1 = [tri[1][0]-tri[0][0], tri[1][1]-tri[0][1], tri[1][2]-tri[0][2]];
    let e2 = [tri[2][0]-tri[0][0], tri[2][1]-tri[0][1], tri[2][2]-tri[0][2]];
    let e1_len = (e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]).sqrt();
    let e2_len = (e2[0]*e2[0]+e2[1]*e2[1]+e2[2]*e2[2]).sqrt();
    if e1_len < 0.01 || e2_len < 0.01 { return vec![(tri[0], n)]; }

    let n_u = (e1_len / coarse_spacing).ceil() as usize;
    let n_v = (e2_len / coarse_spacing).ceil() as usize;
    let mut out: Vec<([f32;3],[f32;3])> = Vec::new();

    let bary = |u: f32, v: f32| -> [f32;3] {
        let w = 1.0 - u - v;
        [u*tri[0][0]+v*tri[1][0]+w*tri[2][0],
         u*tri[0][1]+v*tri[1][1]+w*tri[2][1],
         u*tri[0][2]+v*tri[1][2]+w*tri[2][2]]
    };

    for iu in 0..n_u {
        for iv in 0..n_v {
            let u0 = iu as f32 / n_u as f32;
            let v0 = iv as f32 / n_v as f32;
            if u0 + v0 > 1.0 { continue; }
            let cu = u0 + 0.5 / n_u as f32;
            let cv = v0 + 0.5 / n_v as f32;
            if cu + cv <= 1.0 {
                out.push((bary(cu, cv), n));
            }
            if coarse_spacing > fine_spacing * 1.5 {
                let edge_near = u0 < 0.3 || v0 < 0.3 || u0 > 0.7 || v0 > 0.7
                    || (1.0 - u0 - v0) < 0.3;
                if edge_near && coarse_spacing > fine_spacing * 2.0 {
                    let sub_n = (coarse_spacing / fine_spacing).ceil() as usize;
                    for si in 0..sub_n {
                        for sj in 0..sub_n {
                            let su = u0 + (si as f32 + 0.5) / (n_u as f32 * sub_n as f32);
                            let sv = v0 + (sj as f32 + 0.5) / (n_v as f32 * sub_n as f32);
                            if su + sv <= 1.0 {
                                out.push((bary(su, sv), n));
                            }
                        }
                    }
                }
            }
        }
    }
    if out.is_empty() { out.push((tri[0], n)); }
    out
}

#[allow(dead_code)]
fn sample_triangle_surface(tri: &[[f32;3];3], spacing: f32) -> Vec<([f32;3], [f32;3])> {
    let n = tri_normal(tri);
    sample_triangle_adaptive(tri, n, spacing, spacing * 4.0)
}

// ── Fixed sphere-sample directions (octahedron vertices) ──

// Two alternating sets of 8 directions each. Combined they cover the 14
// unique octahedron + cube-vertex directions (2 overlap: ±Z).
// Alternating per segment keeps per-segment rays low while still covering
// all angles across neighboring segments.

const SPHERE_DIRS_A: [[f32;3]; 8] = [
    // 6 octahedron (axis-aligned)
    [ 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.0], [ 0.0,-1.0, 0.0],
    [ 0.0, 0.0, 1.0], [ 0.0, 0.0,-1.0],
    // 2 cube diagonals
    [ 0.57735, 0.57735, 0.57735],
    [ 0.57735, 0.57735,-0.57735],
];

const SPHERE_DIRS_B: [[f32;3]; 8] = [
    // 2 axis-aligned (overlap with A for vertical coverage)
    [ 0.0, 0.0, 1.0], [ 0.0, 0.0,-1.0],
    // 6 remaining cube diagonals
    [ 0.57735,-0.57735, 0.57735],
    [ 0.57735,-0.57735,-0.57735],
    [-0.57735, 0.57735, 0.57735],
    [-0.57735, 0.57735,-0.57735],
    [-0.57735,-0.57735, 0.57735],
    [-0.57735,-0.57735,-0.57735],
];

// ── Inside-volume test ──

fn ray_tri_intersect_x(p: [f32;3], tri: &[[f32;3];3]) -> Option<f32> {
    let e1=[tri[1][0]-tri[0][0],tri[1][1]-tri[0][1],tri[1][2]-tri[0][2]];
    let e2=[tri[2][0]-tri[0][0],tri[2][1]-tri[0][1],tri[2][2]-tri[0][2]];
    let s=[p[0]-tri[0][0],p[1]-tri[0][1],p[2]-tri[0][2]];
    let cp=[0.0, -e2[2], e2[1]];
    let det=e1[0]*cp[0]+e1[1]*cp[1]+e1[2]*cp[2];
    if det.abs()<1e-12{return None;}
    let inv=1.0/det;
    let u=(s[0]*cp[0]+s[1]*cp[1]+s[2]*cp[2])*inv;
    if u<0.0||u>1.0{return None;}
    let qp=[s[1]*e1[2]-s[2]*e1[1],s[2]*e1[0]-s[0]*e1[2],s[0]*e1[1]-s[1]*e1[0]];
    let v=qp[0]*inv;
    if v<0.0||u+v>1.0{return None;}
    let tt=(e2[0]*qp[0]+e2[1]*qp[1]+e2[2]*qp[2])*inv;
    if tt>1e-6{Some(tt)}else{None}
}

fn count_intersections(p: [f32;3], mesh: &STLMesh, nodes: &[BVHNode], tri_ps: &[TriP2]) -> u32 {
    let ray = Ray::new(Vec3::from(p), Vec3::from([1.0,0.0,0.0]));
    let mut cnt = 0u32;
    let mut stack = [0usize; 64];
    let mut sp = 1usize;
    stack[0] = 0;
    while sp > 0 {
        sp -= 1;
        match nodes[stack[sp]] {
            BVHNode::Node { child_l_aabb, child_l_index, child_r_aabb, child_r_index, .. } => {
                if ray.intersects_aabb(&child_l_aabb) {
                    stack[sp] = child_l_index;
                    sp += 1;
                }
                if ray.intersects_aabb(&child_r_aabb) {
                    stack[sp] = child_r_index;
                    sp += 1;
                }
            }
            BVHNode::Leaf { shape_index, .. } => {
                let ti = tri_ps[shape_index].idx;
                if let Some(_) = ray_tri_intersect_x(p, &mesh.triangles[ti]) {
                    cnt += 1;
                }
            }
        }
    }
    cnt
}

// ── Public API ──

pub fn cull_with_rays(
    segments: &mut [Segment], mesh: &STLMesh, max_width: f32,
    gc_center: [f32;3], stl_center: [f32;3], swap_xy: bool,
) {
    let t0=std::time::Instant::now();
    let xform=|x:f32,y:f32,z:f32|->[f32;3]{
        if swap_xy{[stl_center[0]+(y-gc_center[1]),stl_center[1]-(x-gc_center[0]),z-(gc_center[2]-stl_center[2])]}
        else{[x-(gc_center[0]-stl_center[0]),y-(gc_center[1]-stl_center[1]),z-(gc_center[2]-stl_center[2])]}
    };
    let stl_segs: Vec<Segment> = segments.iter().map(|s| {
        let mut out = s.clone();
        let stl_s = xform(s.data[0], s.data[1], s.data[2]);
        let stl_e = xform(s.data[3], s.data[4], s.data[5]);
        out.data[0]=stl_s[0];out.data[1]=stl_s[1];out.data[2]=stl_s[2];
        out.data[3]=stl_e[0];out.data[4]=stl_e[1];out.data[5]=stl_e[2];
        out
    }).collect();
    let grid = SegBVH::from_segments(&stl_segs);
    let build_seg_ms = t0.elapsed();
    let sphere_r = 3.0;
    let hit_r = max_width * 0.3;
    let kept: Vec<AtomicBool> = (0..stl_segs.len()).map(|_| AtomicBool::new(false)).collect();
    let tr = AtomicU32::new(0);
    let t1 = std::time::Instant::now();

    let mut tri_ps: Vec<TriP2> = mesh.triangles.iter().enumerate().map(|(i,t)| TriP2{idx:i, v:*t, ni:0}).collect();
    let tri_bvh = std::sync::Arc::new(BVH::build(&mut tri_ps));
    let tri_ps = std::sync::Arc::new(tri_ps);
    let tri_nodes = tri_bvh.nodes.clone();

    let kept = std::sync::Arc::new(kept);
    let tr = std::sync::Arc::new(tr);
    let stl_segs = std::sync::Arc::new(stl_segs);
    let grid = std::sync::Arc::new(grid);
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk_size = (stl_segs.len() + n_threads - 1) / n_threads;

    std::thread::scope(|s| {
        for chunk_start in (0..stl_segs.len()).step_by(chunk_size) {
            let start = chunk_start;
            let end = (start + chunk_size).min(stl_segs.len());
            let sg = stl_segs.clone();
            let gr = grid.clone();
            let kp = kept.clone();
            let trc = tr.clone();
            let nodes = tri_nodes.clone();
            let tps = tri_ps.clone();
            s.spawn(move || {
                for i in start..end {
                    if kp[i].load(Ordering::Relaxed) { continue; }
                    if !sg[i].is_model() { continue; }
                    if sg[i].role == Role::InternalInfill as u8 { continue; }
                    let sgg = &sg[i];
                    let mx = (sgg.data[0]+sgg.data[3])*0.5;
                    let my = (sgg.data[1]+sgg.data[4])*0.5;
                    let mz = (sgg.data[2]+sgg.data[5])*0.5;
                    let dirs = if i % 2 == 0 { &SPHERE_DIRS_A } else { &SPHERE_DIRS_B };
                    for &off in dirs {
                        let px = mx+off[0]*sphere_r; let py = my+off[1]*sphere_r; let pz = mz+off[2]*sphere_r;
                        if count_intersections([px,py,pz], &mesh, &nodes, &tps) % 2 != 0 { continue; }
                        let rdx = mx-px; let rdy = my-py; let rdz = mz-pz;
                        let rl = (rdx*rdx+rdy*rdy+rdz*rdz).sqrt().max(1e-12);
                        trc.fetch_add(1, Ordering::Relaxed);
                        if let Some((li,_)) = gr.intersect([px,py,pz],[rdx/rl,rdy/rl,rdz/rl],hit_r) {
                            kp[li].store(true, Ordering::Relaxed);
                        }
                    }
                    for j in 0..4 {
                        let a = j as f32 * std::f32::consts::FRAC_PI_4;
                        let px = mx+sphere_r*a.cos(); let py = my+sphere_r*a.sin(); let pz = mz;
                        if count_intersections([px,py,pz], &mesh, &nodes, &tps) % 2 != 0 { continue; }
                        let rl = sphere_r.max(1e-12);
                        trc.fetch_add(1, Ordering::Relaxed);
                        if let Some((li,_)) = gr.intersect([px,py,pz],[(mx-px)/rl,(my-py)/rl,0.0],hit_r) {
                            kp[li].store(true, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });
    let mut kept: Vec<bool> = std::sync::Arc::try_unwrap(kept).unwrap().iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let total_rays = std::sync::Arc::try_unwrap(tr).unwrap().load(Ordering::Relaxed);
    let query_ms = t1.elapsed();
    let gap_ms = std::time::Instant::now();

    let gap_n = 3usize;
    let seg_len = segments.len();
    for _ in 0..2 {
        let mut fill = Vec::new();
        for i in 0..seg_len {
            if kept[i] { continue; }
            let mut left = false;
            for j in i.saturating_sub(gap_n)..i {
                if kept[j] { left = true; break; }
            }
            let mut right = false;
            for j in (i+1)..seg_len.min(i+1+gap_n) {
                if kept[j] { right = true; break; }
            }
            if left && right { fill.push(i); }
        }
        for &i in &fill { kept[i] = true; }
    }

    let mut culled = 0usize;
    for (i, seg) in segments.iter_mut().enumerate() {
        if !kept[i] { seg.set_width(0.0); culled += 1; }
    }
    let gap_us = gap_ms.elapsed().as_micros();
    eprintln!("  Ray cull: {} rays, {} / {} culled (ray={}ms, seg_bvh={}ms, gap={})",
        total_rays, culled, segments.len(), query_ms.as_millis(), build_seg_ms.as_millis(),
        if gap_us >= 1000 { format!("{}ms", gap_us / 1000) } else { format!("{}µs", gap_us) });
}
