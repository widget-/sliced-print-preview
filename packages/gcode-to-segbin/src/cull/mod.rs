mod contour;
mod ray;

pub use self::contour::cull_with_contours;
pub use self::ray::cull_with_rays;

use crate::segment::Segment;

/// Compute the bounding box of all model (non-travel) segments in G-code space.
pub fn compute_gcode_bbox(segments: &[Segment]) -> ([f32; 3], [f32; 3]) {
    let mut bmin = [f32::MAX; 3];
    let mut bmax = [f32::MIN; 3];
    for seg in segments {
        if !seg.is_model() { continue; }
        for i in 0..3 {
            bmin[i] = bmin[i].min(seg.data[i]);
            bmax[i] = bmax[i].max(seg.data[i]);
            bmin[i] = bmin[i].min(seg.data[3 + i]);
            bmax[i] = bmax[i].max(seg.data[3 + i]);
        }
    }
    (bmin, bmax)
}
