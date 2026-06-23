use crate::segment::{Role, Segment};

pub struct GCodeParser {
    pos: [f64; 3],
    relative_xyz: bool,
    relative_e: bool,
    imperial: bool,
    extruder: f64,
    layer_z: f32,
    role: u8,
    width: f32,
    pub segments: Vec<Segment>,
    pub line_count: u64,
}

impl GCodeParser {
    pub fn new() -> Self {
        Self {
            pos: [0.0; 3],
            relative_xyz: false,
            relative_e: false,
            imperial: false,
            extruder: 0.0,
            layer_z: 0.0,
            role: Role::Other as u8,
            width: 0.4,
            segments: Vec::with_capacity(128_000),
            line_count: 0,
        }
    }

    pub fn parse_line(&mut self, raw: &str) {
        self.line_count += 1;
        let line = raw.trim();
        if line.is_empty() { return; }

        if let Some(c) = line.strip_prefix(';') {
            self.parse_comment(c);
            return;
        }

        let code = if let Some(sc_pos) = line.find(';') {
            &line[..sc_pos].trim()
        } else {
            line
        };
        if code.is_empty() { return; }

        let letter = code.as_bytes()[0] as char;
        if letter != 'G' && letter != 'M' { return; }

        let num_str = code[1..].split_whitespace().next().unwrap_or("");
        let num: i32 = match num_str.parse() {
            Ok(n) => n,
            Err(_) => return,
        };

        let params = self.parse_params(code);

        match letter {
            'G' => self.handle_g(num, &params),
            'M' => self.handle_m(num, &params),
            _ => {}
        }
    }

    fn parse_params<'a>(&self, line: &'a str) -> Vec<(char, f64)> {
        let mut params = Vec::with_capacity(8);
        let mut chars = line.chars().peekable();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_alphabetic() {
                let letter = c;
                chars.next();
                let num_str: String = chars.by_ref().take_while(|&c| {
                    c == '.' || c == '-' || c.is_ascii_digit()
                }).collect();
                if let Ok(val) = num_str.parse::<f64>() {
                    params.push((letter, val));
                }
            } else {
                chars.next();
            }
        }
        params
    }

    fn get_param(&self, params: &[(char, f64)], key: char) -> Option<f64> {
        params.iter().find(|&&(k, _)| k == key).map(|&(_, v)| v)
    }

    fn handle_g(&mut self, num: i32, params: &[(char, f64)]) {
        match num {
            0 | 1 => self.do_linear_move(num == 0, params),
            2 | 3 => self.do_arc_move(num == 2, params),
            90 => self.relative_xyz = false,
            91 => self.relative_xyz = true,
            92 => self.do_set_position(params),
            20 => self.imperial = true,
            21 => self.imperial = false,
            _ => {}
        }
    }

    fn handle_m(&mut self, num: i32, _params: &[(char, f64)]) {
        match num {
            82 => self.relative_e = false,
            83 => self.relative_e = true,
            _ => {}
        }
    }

    fn to_mm(&self, val: f64) -> f64 {
        if self.imperial { val * 25.4 } else { val }
    }

    fn do_linear_move(&mut self, _is_travel: bool, params: &[(char, f64)]) {
        let mut new_pos = self.pos;
        if let Some(x) = self.get_param(params, 'X') {
            new_pos[0] = if self.relative_xyz { self.pos[0] + self.to_mm(x) } else { self.to_mm(x) };
        }
        if let Some(y) = self.get_param(params, 'Y') {
            new_pos[1] = if self.relative_xyz { self.pos[1] + self.to_mm(y) } else { self.to_mm(y) };
        }
        if let Some(z) = self.get_param(params, 'Z') {
            new_pos[2] = if self.relative_xyz { self.pos[2] + self.to_mm(z) } else { self.to_mm(z) };
        }

        let has_e = self.get_param(params, 'E').is_some();
        let is_travel = !has_e;
        if is_travel {
            self.pos = new_pos;
            return;
        }

        let sx = self.pos[0] as f32;
        let sy = self.pos[1] as f32;
        let sz = self.pos[2] as f32;
        let ex = new_pos[0] as f32;
        let ey = new_pos[1] as f32;
        let ez = new_pos[2] as f32;

        let dx = ex - sx;
        let dy = ey - sy;
        let dz = ez - sz;
        if dx * dx + dy * dy + dz * dz < 0.0001 {
            self.pos = new_pos;
            return;
        }

        let mut seg = Segment::line(sx, sy, sz, ex, ey, ez, self.width, self.layer_z);
        seg.role = self.role;
        self.segments.push(seg);
        self.layer_z = new_pos[2] as f32;
        self.pos = new_pos;
    }

    fn do_arc_move(&mut self, _clockwise: bool, params: &[(char, f64)]) {
        let mut new_pos = self.pos;
        if let Some(x) = self.get_param(params, 'X') {
            new_pos[0] = if self.relative_xyz { self.pos[0] + self.to_mm(x) } else { self.to_mm(x) };
        }
        if let Some(y) = self.get_param(params, 'Y') {
            new_pos[1] = if self.relative_xyz { self.pos[1] + self.to_mm(y) } else { self.to_mm(y) };
        }
        if let Some(z) = self.get_param(params, 'Z') {
            new_pos[2] = if self.relative_xyz { self.pos[2] + self.to_mm(z) } else { self.to_mm(z) };
        }

        let has_e = self.get_param(params, 'E').is_some();
        if !has_e { self.pos = new_pos; return; }

        let n_segs = 4.max(1);
        for t in 1..=n_segs {
            let frac = t as f64 / n_segs as f64;
            let px = self.pos[0] + (new_pos[0] - self.pos[0]) * frac;
            let py = self.pos[1] + (new_pos[1] - self.pos[1]) * frac;
            let pz = self.pos[2] + (new_pos[2] - self.pos[2]) * frac;

            if t == 1 {
                let mut seg = Segment::line(
                    self.pos[0] as f32, self.pos[1] as f32, self.pos[2] as f32,
                    px as f32, py as f32, pz as f32,
                    self.width, self.layer_z,
                );
                seg.role = self.role;
                self.segments.push(seg);
            } else {
                let prev = self.segments.last().unwrap();
                let mut seg = Segment::line(
                    prev.data[3], prev.data[4], prev.data[5],
                    px as f32, py as f32, pz as f32,
                    self.width, self.layer_z,
                );
                seg.role = self.role;
                self.segments.push(seg);
            }
        }
        self.pos = new_pos;
    }

    fn do_set_position(&mut self, params: &[(char, f64)]) {
        if let Some(x) = self.get_param(params, 'X') {
            self.pos[0] = self.to_mm(x);
        }
        if let Some(y) = self.get_param(params, 'Y') {
            self.pos[1] = self.to_mm(y);
        }
        if let Some(z) = self.get_param(params, 'Z') {
            self.pos[2] = self.to_mm(z);
        }
        if let Some(e) = self.get_param(params, 'E') {
            self.extruder = e;
        }
    }

    fn parse_comment(&mut self, comment: &str) {
        let c = comment.trim();
        if let Some(layer_str) = c.strip_prefix("LAYER:") {
            if let Ok(z) = layer_str.trim().parse::<f32>() {
                self.layer_z = z;
            }
            return;
        }
        if let Some(type_str) = c.strip_prefix("TYPE:").or_else(|| c.strip_prefix("FEATURE:")) {
            self.role = Role::from_str(type_str.trim()) as u8;
            return;
        }
        if let Some(width_str) = c.strip_prefix("WIDTH:").or_else(|| c.strip_prefix("LINE_WIDTH:")) {
            if let Ok(w) = width_str.trim().parse::<f32>() {
                self.width = w;
            }
            return;
        }
        if let Some(layer_str) = c.strip_prefix("LAYER") {
            if let Some(n_str) = layer_str.trim().strip_prefix(':') {
                if let Ok(z) = n_str.parse::<f32>() {
                    self.layer_z = z;
                }
            }
        }
    }
}
