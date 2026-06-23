//! gcode-to-segbin — Convert GCode from OrcaSlicer / PrusaSlicer / SuperSlicer
//! into a compact binary segment file (.segbin) for GPU-instanced rendering.
//!
//! Usage:  gcode-to-segbin input.gcode output.segbin [model.stl] [--cull-method ray]

mod segment;
mod parser;
mod writer;
mod stl;
mod cull;
mod arcs;

use std::{env, fs, process};
use segment::Role;

fn run() -> Result<(), String> {
    let started = std::time::Instant::now();
    let args: Vec<String> = env::args().collect();
    let usage = || {
        eprintln!("Usage: {} <input.gcode> <output.segbin> [model.stl]", args.first().map(|s| s.as_str()).unwrap_or("gcode-to-segbin"));
        eprintln!();
        eprintln!("Converts GCode from OrcaSlicer/PrusaSlicer/SuperSlicer into .segbin");
        eprintln!("binary format for GPU-instanced Three.js rendering.");
        eprintln!();
        eprintln!("Optional model.stl enables surface culling — segments far from the");
        eprintln!("model surface (buried infill) are zero-width and render as invisible.");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  gcode-to-segbin benchy.gcode benchy.segbin");
        eprintln!("  gcode-to-segbin benchy.gcode benchy.segbin benchy.stl");
    };
    if args.len() < 3 {
        usage();
        return Err("Missing arguments".to_string());
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let stl_path = args.get(3).map(|s| s.as_str());
    let use_ray = args.iter().any(|a| a == "--cull-method=ray")
        || args.windows(2).any(|w| w[0] == "--cull-method" && w[1] == "ray");

    eprintln!("Reading: {}", input_path);
    let gcode = fs::read_to_string(input_path)
        .map_err(|e| format!("Failed to read {}: {}", input_path, e))?;

    let file_size_kb = gcode.len() as f64 / 1024.0;
    eprintln!("  {} KB GCode", (file_size_kb * 10.0).round() / 10.0);

    let mut parser = parser::GCodeParser::new();

    let t0 = std::time::Instant::now();
    for line in gcode.lines() {
        parser.parse_line(line);
    }
    let elapsed = t0.elapsed();

    let seg_count = parser.segments.len();
    let merged = parser.line_count - seg_count as u64;
    eprintln!("Parsed {} lines → {} segments ({} merged, {:.2?})",
        parser.line_count, seg_count, merged, elapsed);

    writer::compute_chain_flags(&mut parser.segments, 45.0);

    if let Some(stl) = stl_path {
        eprintln!("Reading model: {}", stl);
        let mesh = stl::parse_stl(stl)?;
        eprintln!("  {} triangles", mesh.triangles.len());

        let (gc_min, gc_max) = cull::compute_gcode_bbox(&parser.segments);
        let (stl_min, stl_max) = stl::compute_stl_bbox(&mesh);

        let gc_sx = gc_max[0] - gc_min[0];
        let gc_sy = gc_max[1] - gc_min[1];
        let stl_sx = stl_max[0] - stl_min[0];
        let stl_sy = stl_max[1] - stl_min[1];
        let swap_xy = (gc_sx - stl_sy).abs() < (gc_sx - stl_sx).abs()
            && (gc_sy - stl_sx).abs() < (gc_sy - stl_sy).abs();
        eprintln!("  GCode center: ({:.1}, {:.1}, {:.1})",
            (gc_min[0] + gc_max[0]) * 0.5,
            (gc_min[1] + gc_max[1]) * 0.5,
            (gc_min[2] + gc_max[2]) * 0.5);
        eprintln!("  STL center:   ({:.1}, {:.1}, {:.1})",
            (stl_min[0] + stl_max[0]) * 0.5,
            (stl_min[1] + stl_max[1]) * 0.5,
            (stl_min[2] + stl_max[2]) * 0.5);
        eprintln!("  Axes swapped: {}", swap_xy);
        let gc_center = [(gc_min[0] + gc_max[0]) * 0.5, (gc_min[1] + gc_max[1]) * 0.5, (gc_min[2] + gc_max[2]) * 0.5];
        let stl_center = [(stl_min[0] + stl_max[0]) * 0.5, (stl_min[1] + stl_max[1]) * 0.5, (stl_min[2] + stl_max[2]) * 0.5];

        let mut max_width = 0.0f32;
        let mut max_layer_h = 0.0f32;
        let mut prev_z = 0.0f32;
        for (i, seg) in parser.segments.iter().enumerate() {
            if seg.role >= Role::Travel as u8 { continue; }
            max_width = max_width.max(seg.data[6]);
            if i > 0 {
                let dz = (seg.data[7] - prev_z).abs();
                if dz > 0.001 { max_layer_h = max_layer_h.max(dz); }
            }
            prev_z = seg.data[7];
        }
        eprintln!("  Max line width: {:.3}mm  layer h: {:.3}mm", max_width, max_layer_h);

        if use_ray {
            eprintln!("  Cull method: ray-casting");
            cull::cull_with_rays(&mut parser.segments, &mesh, max_width, gc_center, stl_center, swap_xy);
        } else {
            eprintln!("  Cull method: contour");
            cull::cull_with_contours(&mut parser.segments, &mesh, max_width, max_layer_h, gc_center, stl_center, swap_xy);
        }
    }

    let arc_start = std::time::Instant::now();
    let before_arc = parser.segments.len();
    arcs::apply_arc_subdivision(&mut parser.segments, 10.0, 3.0);
    let arc_count = parser.segments.len() - before_arc;
    let arc_ms = arc_start.elapsed();
    if arc_count > 0 {
        eprintln!("  Arc subdivision: +{} segments at joints ({}ms)", arc_count, arc_ms.as_millis());
    }

    writer::write_segbin(output_path, &parser.segments)?;
    eprintln!("  Total: {}ms", started.elapsed().as_millis());

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
