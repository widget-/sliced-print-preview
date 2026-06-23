#[derive(Debug, Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
pub enum Role {
    Perimeter = 0,
    ExternalPerimeter = 1,
    OverhangPerimeter = 2,
    InternalInfill = 3,
    SolidInfill = 4,
    TopSolidInfill = 5,
    BottomSurface = 6,
    BridgeInfill = 7,
    InternalBridgeInfill = 8,
    Travel = 9,
    SkirtBrim = 10,
    Support = 11,
    Ironing = 12,
    Other = 13,
}

impl Role {
    pub fn from_str(s: &str) -> Self {
        match s.trim() {
            "External perimeter" | "ExternalPerimeter" => Self::ExternalPerimeter,
            "Overhang perimeter" | "OverhangPerimeter" => Self::OverhangPerimeter,
            "Internal infill" | "InternalInfill" => Self::InternalInfill,
            "Solid infill" | "SolidInfill" | "Internal solid infill" | "InternalSolidInfill" => Self::SolidInfill,
            "Top solid infill" | "TopSolidInfill" => Self::TopSolidInfill,
            "Bottom surface" | "BottomSurface" => Self::BottomSurface,
            "Bridge infill" | "BridgeInfill" => Self::BridgeInfill,
            "Internal bridge infill" | "InternalBridgeInfill" => Self::InternalBridgeInfill,
            "Outer wall" | "OuterWall" => Self::ExternalPerimeter,
            "Inner wall" | "InnerWall" => Self::Perimeter,
            "Sparse infill" => Self::InternalInfill,
            "Overhang wall" => Self::OverhangPerimeter,
            "Top surface" => Self::TopSolidInfill,
            "Internal Bridge" => Self::InternalBridgeInfill,
            "Bridge" => Self::BridgeInfill,
            "Brim" => Self::SkirtBrim,
            "Skirt" | "Skirt/Brim" => Self::SkirtBrim,
            "Support" | "Support material" | "SupportMaterial" => Self::Support,
            "Ironing" => Self::Ironing,
            "Travel" | "Move" => Self::Travel,
            _ => Self::Other,
        }
    }

    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            Self::Perimeter => "Perimeter",
            Self::ExternalPerimeter => "ExternalPerimeter",
            Self::OverhangPerimeter => "OverhangPerimeter",
            Self::InternalInfill => "InternalInfill",
            Self::SolidInfill => "SolidInfill",
            Self::TopSolidInfill => "TopSolidInfill",
            Self::BottomSurface => "BottomSurface",
            Self::BridgeInfill => "BridgeInfill",
            Self::InternalBridgeInfill => "InternalBridgeInfill",
            Self::Travel => "Travel",
            Self::SkirtBrim => "SkirtBrim",
            Self::Support => "Support",
            Self::Ironing => "Ironing",
            Self::Other => "Other",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub seg_type: u8,
    pub data: [f32; 8],
    pub role: u8,
    pub chain_continue: u8,
}

#[allow(dead_code)]
impl Segment {
    pub fn line(sx: f32, sy: f32, sz: f32, ex: f32, ey: f32, ez: f32, width: f32, layer_z: f32) -> Self {
        Self { seg_type: 0, data: [sx, sy, sz, ex, ey, ez, width, layer_z], role: Role::Other as u8, chain_continue: 0 }
    }

    /// Arc segment: data = [p0x,p0y,p0z, p1x,p1y,p1z, width, packed]
    /// where packed = layer_z + weight * 0.0001
    /// P0 = tangent in, P1 = corner control point, width = extrusion width.
    /// Weight (conic sharpness, ~2-3) is decoded in the frontend.
    /// P2 (tangent out) is read from the next segment's start on the GPU.
    pub fn arc(p0: [f32;3], p1: [f32;3], width: f32, weight: f32, layer_z: f32) -> Self {
        Self {
            seg_type: 1,
            data: [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], width, layer_z + weight * 0.0001],
            role: Role::Other as u8,
            chain_continue: 0,
        }
    }

    pub fn is_model(&self) -> bool {
        self.role < Role::Travel as u8
    }

    pub fn start(&self) -> [f32; 3] {
        [self.data[0], self.data[1], self.data[2]]
    }

    pub fn end(&self) -> [f32; 3] {
        [self.data[3], self.data[4], self.data[5]]
    }

    pub fn midpoint(&self) -> [f32; 3] {
        [(self.data[0] + self.data[3]) * 0.5,
         (self.data[1] + self.data[4]) * 0.5,
         (self.data[2] + self.data[5]) * 0.5]
    }

    pub fn width(&self) -> f32 { self.data[6] }
    pub fn layer_z(&self) -> f32 { self.data[7] }

    pub fn set_width(&mut self, w: f32) { self.data[6] = w; }
}
