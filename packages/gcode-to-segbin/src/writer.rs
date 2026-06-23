use std::fs;
use crate::segment::Segment;

pub fn compute_chain_flags(segments: &mut [Segment], angle_deg: f32) {
    let angle_thresh = angle_deg.to_radians().cos();
    let tol = 0.01;
    for i in 0..segments.len().saturating_sub(1) {
        let (role_a, role_b) = (segments[i].role, segments[i + 1].role);
        let width_a = segments[i].data[6];
        let width_b = segments[i + 1].data[6];
        let layer_a = segments[i].data[7];
        let layer_b = segments[i + 1].data[7];
        if role_a != role_b || width_a != width_b || layer_a != layer_b { continue; }
        let dx = segments[i].data[3] - segments[i + 1].data[0];
        let dy = segments[i].data[4] - segments[i + 1].data[1];
        let dz = segments[i].data[5] - segments[i + 1].data[2];
        if dx * dx + dy * dy + dz * dz > tol * tol { continue; }
        let adx = segments[i].data[3] - segments[i].data[0];
        let ady = segments[i].data[4] - segments[i].data[1];
        let adz = segments[i].data[5] - segments[i].data[2];
        let a_len_sq = adx * adx + ady * ady + adz * adz;
        if a_len_sq < 0.0001 { continue; }
        let a_len = a_len_sq.sqrt();
        let bdx = segments[i + 1].data[3] - segments[i + 1].data[0];
        let bdy = segments[i + 1].data[4] - segments[i + 1].data[1];
        let bdz = segments[i + 1].data[5] - segments[i + 1].data[2];
        let b_len_sq = bdx * bdx + bdy * bdy + bdz * bdz;
        if b_len_sq < 0.0001 { continue; }
        let b_len = b_len_sq.sqrt();
        let dot = (adx * bdx + ady * bdy + adz * bdz) / (a_len * b_len);
        if dot >= angle_thresh {
            segments[i].chain_continue = 1;
        }
    }
}

pub fn write_segbin(path: &str, segments: &[Segment]) -> Result<(), String> {
    let count = segments.len();
    let float_data_len = count * 8 * 4;
    let role_data_len = count;
    let chain_data_len = count;
    let seg_type_len = count;

    let mut buf = Vec::with_capacity(
        16 + float_data_len + role_data_len + chain_data_len + seg_type_len,
    );
    buf.extend_from_slice(&0x31474553u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&(count as u32).to_le_bytes());
    buf.extend_from_slice(&(7u32 | (1 << 3)).to_le_bytes());

    // Float data: 8 × f32 per segment
    for seg in segments {
        for val in &seg.data {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }

    // Role data
    for seg in segments {
        buf.push(seg.role);
    }

    // Chain data
    for seg in segments {
        buf.push(seg.chain_continue);
    }

    // Seg type data
    for seg in segments {
        buf.push(seg.seg_type);
    }

    fs::write(path, &buf).map_err(|e| format!("Failed to write {}: {}", path, e))?;

    let mb = buf.len() as f64 / (1024.0 * 1024.0);
    eprintln!("  Wrote {} segments ({:.2} MB) to {}", count, mb, path);
    Ok(())
}
