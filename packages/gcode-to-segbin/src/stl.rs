use std::fs;

pub struct STLMesh {
    pub triangles: Vec<[[f32; 3]; 3]>,
}

fn parse_stl_binary(data: &[u8], count: usize) -> Result<STLMesh, String> {
    let mut triangles = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 50;
        let a = [
            f32::from_le_bytes([data[off + 12], data[off + 13], data[off + 14], data[off + 15]]),
            f32::from_le_bytes([data[off + 16], data[off + 17], data[off + 18], data[off + 19]]),
            f32::from_le_bytes([data[off + 20], data[off + 21], data[off + 22], data[off + 23]]),
        ];
        let b = [
            f32::from_le_bytes([data[off + 24], data[off + 25], data[off + 26], data[off + 27]]),
            f32::from_le_bytes([data[off + 28], data[off + 29], data[off + 30], data[off + 31]]),
            f32::from_le_bytes([data[off + 32], data[off + 33], data[off + 34], data[off + 35]]),
        ];
        let c = [
            f32::from_le_bytes([data[off + 36], data[off + 37], data[off + 38], data[off + 39]]),
            f32::from_le_bytes([data[off + 40], data[off + 41], data[off + 42], data[off + 43]]),
            f32::from_le_bytes([data[off + 44], data[off + 45], data[off + 46], data[off + 47]]),
        ];
        triangles.push([a, b, c]);
    }
    Ok(STLMesh { triangles })
}

fn parse_stl_ascii(content: &str) -> Result<STLMesh, String> {
    let mut triangles = Vec::new();
    let mut verts: Vec<[f32; 3]> = Vec::new();
    for line in content.lines() {
        let t = line.trim();
        if t.starts_with("vertex ") {
            let parts: Vec<&str> = t[7..].split_whitespace().collect();
            if parts.len() >= 3 {
                let x = parts[0].parse::<f32>().map_err(|_| "Bad STL vertex".to_string())?;
                let y = parts[1].parse::<f32>().map_err(|_| "Bad STL vertex".to_string())?;
                let z = parts[2].parse::<f32>().map_err(|_| "Bad STL vertex".to_string())?;
                verts.push([x, y, z]);
            }
        } else if t.starts_with("endloop") {
            if verts.len() >= 3 {
                triangles.push([verts[0], verts[1], verts[2]]);
            }
            verts.clear();
        }
    }
    if triangles.is_empty() {
        return Err("No triangles found in ASCII STL".to_string());
    }
    Ok(STLMesh { triangles })
}

pub fn parse_stl(path: &str) -> Result<STLMesh, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to read STL {}: {}", path, e))?;
    let file_len = data.len();
    if file_len < 84 {
        return Err("STL file too small".to_string());
    }
    let count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if file_len == 84 + count * 50 {
        return parse_stl_binary(&data[84..], count);
    }
    let content = std::str::from_utf8(&data)
        .map_err(|_| "STL is not valid UTF-8 (not binary STL)".to_string())?;
    parse_stl_ascii(content)
}

pub fn compute_stl_bbox(mesh: &STLMesh) -> ([f32; 3], [f32; 3]) {
    let mut bmin = [f32::MAX; 3];
    let mut bmax = [f32::MIN; 3];
    for tri in &mesh.triangles {
        for v in tri {
            for d in 0..3 {
                bmin[d] = bmin[d].min(v[d]);
                bmax[d] = bmax[d].max(v[d]);
            }
        }
    }
    (bmin, bmax)
}
