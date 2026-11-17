use springrep::{
    blend::fillet,
    boolean::difference,
    math::{Frame, Vec2, Vec3},
    measure,
    profile::Profile,
    step,
    sweep::extrude,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X)?;
    let below = Frame::new(Vec3::new(0.0, 0.0, -1.0), Vec3::Z, Vec3::X)?;

    let plate = extrude(&Profile::rect(Vec2::ZERO, 60.0, 40.0), &base, 8.0, "plate")?;
    let bore = extrude(&Profile::circle(Vec2::ZERO, 12.0), &below, 10.0, "bore")?;

    let part = difference(&plate, &bore)?;
    let part = fillet(&part, &[("plate.end", "bore.side")], 1.5, "round")?;

    println!("volume  {:.6} mm³", measure::volume(&part));
    println!("area    {:.6} mm²", measure::area(&part));
    std::fs::write("bracket.step", step::write(&part, "bracket")?)?;
    println!("wrote bracket.step, {} faces", part.face_count());
    Ok(())
}
