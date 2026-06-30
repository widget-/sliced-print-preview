use crate::segment::{Role, Segment};

fn log_curves_enabled() -> bool {
    std::env::var("GCODE_LOG_CURVES").is_ok()
}

/// Insert arc segments at width‑change boundaries where segments are
/// geometrically continuous but differ in extrusion width.  This avoids
/// expensive endcaps: the arc + per‑segment chain‑width interpolation
/// in the vertex shader handle the transition smoothly.
pub fn apply_width_change_arcs(segments: &mut Vec<Segment>, conic_rho_factor: f32) {
    let tol = 0.01;
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    let mut i = 0;
    while i < segments.len() {
        let has_pair = i + 1 < segments.len()
            && segments[i].data[6] >= 0.001
            && segments[i + 1].data[6] >= 0.001;

        if has_pair
            && (segments[i + 1].data[7] - segments[i].data[7]).abs() <= 0.001
            && (segments[i + 1].data[6] - segments[i].data[6]).abs() > 0.001
        {
            let dx = segments[i].data[3] - segments[i + 1].data[0];
            let dy = segments[i].data[4] - segments[i + 1].data[1];
            let dz = segments[i].data[5] - segments[i + 1].data[2];
            if dx * dx + dy * dy + dz * dz <= tol * tol {
                let adx = segments[i].data[3] - segments[i].data[0];
                let ady = segments[i].data[4] - segments[i].data[1];
                let adz = segments[i].data[5] - segments[i].data[2];
                let a_len_sq = adx*adx+ady*ady+adz*adz;
                let bdx = segments[i+1].data[3] - segments[i+1].data[0];
                let bdy = segments[i+1].data[4] - segments[i+1].data[1];
                let bdz = segments[i+1].data[5] - segments[i+1].data[2];
                let b_len_sq = bdx*bdx+bdy*bdy+bdz*bdz;
                if a_len_sq >= 0.0001 && b_len_sq >= 0.0001 {
                    let a_len = a_len_sq.sqrt();
                    let b_len = b_len_sq.sqrt();
                    let in_dir = [adx/a_len, ady/a_len, adz/a_len];
                    let out_dir = [bdx/b_len, bdy/b_len, bdz/b_len];
                    let dot = (in_dir[0]*out_dir[0]+in_dir[1]*out_dir[1]+in_dir[2]*out_dir[2]).clamp(-1.0,1.0);
                    // Handle all angles — compute arc parameters matching
                    // apply_arc_subdivision but using each segment's own width.
                    if true {
                        let p = [segments[i].data[3], segments[i].data[4], segments[i].data[5]];
                        let width_a = segments[i].data[6];
                        let width_b = segments[i + 1].data[6];
                        let layer = segments[i].data[7];
                        let theta = dot.acos().max(0.01);
                        let theta_half = theta * 0.5;
                        let tan_half = theta_half.tan();
                        let cos_half = theta_half.cos();
                        // Retract each side by its OWN half-width × tan(θ/2)
                        let max_a = 0.49 * a_len;
                        let max_b = 0.49 * b_len;
                        let d_a = (width_a * 0.5 * tan_half).min(max_a);
                        let d_b = (width_b * 0.5 * tan_half).min(max_b);
                        if d_a >= 0.001 && d_b >= 0.001 {
                            let t_in = [p[0]-in_dir[0]*d_a, p[1]-in_dir[1]*d_a, p[2]-in_dir[2]*d_a];
                            let t_out = [p[0]+out_dir[0]*d_b, p[1]+out_dir[1]*d_b, p[2]+out_dir[2]*d_b];
                            let conic_w = cos_half * conic_rho_factor;

                            let mut inc = segments[i].clone();
                            inc.data[3]=t_in[0];inc.data[4]=t_in[1];inc.data[5]=t_in[2];inc.chain_continue=1;
                            out.push(inc);
                            // Single arc — P2 read from next segment's start on GPU.
                            // Width = first segment's width; shader interpolates to width_b.
                            let mut arc_seg = Segment::arc(t_in, p, width_a, conic_w, layer);
                            arc_seg.role = segments[i].role;
                            arc_seg.chain_continue = 1;
                            out.push(arc_seg);
                            let mut outg = segments[i+1].clone();
                            outg.data[0]=t_out[0];outg.data[1]=t_out[1];outg.data[2]=t_out[2];
                            outg.chain_continue = 1;
                            out.push(outg);
                            i += 2;
                            continue;
                        }
                    }
                }
            }
        }
        out.push(segments[i].clone());
        i += 1;
    }
    *segments = out;
}

pub fn apply_arc_subdivision(segments: &mut Vec<Segment>, max_angle_deg: f32, conic_rho_factor: f32) {
    let max_rad = max_angle_deg.to_radians();
    let tol = 0.01;
    let log_curves = log_curves_enabled();
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    let mut i = 0;
    while i < segments.len() {
        let has_pair = i + 1 < segments.len()
            && segments[i].data[6] >= 0.001
            && segments[i + 1].data[6] >= 0.001;
        let mut did_arc = false;

        if has_pair {
            let role_a = segments[i].role;
            if role_a == segments[i + 1].role
                && role_a < Role::Travel as u8
                && (segments[i + 1].data[6] - segments[i].data[6]).abs() <= 0.001
                && (segments[i + 1].data[7] - segments[i].data[7]).abs() <= 0.001
            {
                let dx = segments[i].data[3] - segments[i + 1].data[0];
                let dy = segments[i].data[4] - segments[i + 1].data[1];
                let dz = segments[i].data[5] - segments[i + 1].data[2];
                if dx * dx + dy * dy + dz * dz <= tol * tol {
                    let adx = segments[i].data[3] - segments[i].data[0];
                    let ady = segments[i].data[4] - segments[i].data[1];
                    let adz = segments[i].data[5] - segments[i].data[2];
                    let a_len_sq = adx*adx+ady*ady+adz*adz;
                    let bdx = segments[i+1].data[3] - segments[i+1].data[0];
                    let bdy = segments[i+1].data[4] - segments[i+1].data[1];
                    let bdz = segments[i+1].data[5] - segments[i+1].data[2];
                    let b_len_sq = bdx*bdx+bdy*bdy+bdz*bdz;
                    if a_len_sq >= 0.0001 && b_len_sq >= 0.0001 {
                        let a_len = a_len_sq.sqrt();
                        let b_len = b_len_sq.sqrt();
                        let in_dir = [adx/a_len, ady/a_len, adz/a_len];
                        let out_dir = [bdx/b_len, bdy/b_len, bdz/b_len];
                        let dot = (in_dir[0]*out_dir[0]+in_dir[1]*out_dir[1]+in_dir[2]*out_dir[2]).clamp(-1.0,1.0);
                        if dot < max_rad.cos() {
                            let theta = dot.acos();
                            if theta >= 0.01 {
                                let width = segments[i].data[6];
                                let layer = segments[i].data[7];
                                let theta_half = theta*0.5;
                                let tan_half = theta_half.tan();
                                let cos_half = theta_half.cos();
                                let mut r = width*0.5;
                                let mut d = r*tan_half;
                                let max_d = 0.49*a_len.min(b_len);
                                if d > max_d { d = max_d; r = d/tan_half; }
                                if r >= 0.001 {
                                    let n_angle = ((theta/max_rad).ceil() as usize).max(1);
                                    let n_length = ((4.0*d/r).floor() as usize).max(1);
                                    let n = n_angle.min(n_length);
                                    let p = [segments[i].data[3],segments[i].data[4],segments[i].data[5]];
                                    let t_in = [p[0]-in_dir[0]*d,p[1]-in_dir[1]*d,p[2]-in_dir[2]*d];
                                    let t_out = [p[0]+out_dir[0]*d,p[1]+out_dir[1]*d,p[2]+out_dir[2]*d];
                                    let conic_w = cos_half*conic_rho_factor;
                                    if log_curves {
                                        eprintln!("CURVE\t{:.3}\t{:.3}\t{:.3}\t{}\t{:.1}°\t{:.2}mm\t{:.2}\t{}",
                                            p[0],p[1],p[2],match role_a{0=>"Perimeter",1=>"ExternalPerimeter",2=>"OverhangPerimeter",6=>"BottomSurface",_=>"Perimeter"},
                                            theta.to_degrees(),width,layer,n);
                                    }
                                    let mut inc = segments[i].clone();
                                    inc.data[3]=t_in[0];inc.data[4]=t_in[1];inc.data[5]=t_in[2];inc.chain_continue=1;
                                    out.push(inc);
                                    // Single arc primitive — P2 read from next segment's start on GPU
                                    let mut arc_seg = Segment::arc(t_in, p, width, conic_w, layer);
                                    arc_seg.role = role_a;
                                    arc_seg.chain_continue = 1;
                                    out.push(arc_seg);
                                    let mut outg = segments[i+1].clone();
                                    outg.data[0]=t_out[0];outg.data[1]=t_out[1];outg.data[2]=t_out[2];
                                    out.push(outg);
                                    i += 2;
                                    did_arc = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        if !did_arc { out.push(segments[i].clone()); i += 1; }
    }
    *segments = out;
}
