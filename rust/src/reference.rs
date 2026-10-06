use astro_float::{BigFloat, Consts, Radix, RoundingMode, Sign};

use crate::{ESCAPE_R2, LN2};

const RM: RoundingMode = RoundingMode::ToEven;

/// Computes the reference orbit Z_0..Z_max_iter at the zoom centre with enough
/// binary precision for the deepest frame, rounding each point to f64.
/// Returned as interleaved (re, im) pairs; Z_0 = 0.
pub fn orbit(re: &str, im: &str, max_iter: i32, ln_zoom: f64) -> anyhow::Result<Vec<f64>> {
    let p = (ln_zoom / LN2 + 64.0) as usize;
    let mut cc = Consts::new()?;
    let cr = BigFloat::parse(re, Radix::Dec, p, RM, &mut cc);
    let ci = BigFloat::parse(im, Radix::Dec, p, RM, &mut cc);
    anyhow::ensure!(!cr.is_nan() && !ci.is_nan(), "invalid centre coordinate");

    let mut orbit = Vec::with_capacity(2 * (max_iter as usize + 1));
    orbit.extend([0.0, 0.0]);
    let (mut zr, mut zi) = (BigFloat::from_word(0, p), BigFloat::from_word(0, p));
    for _ in 0..max_iter {
        let zr2 = zr.mul(&zr, p, RM);
        let zi2 = zi.mul(&zi, p, RM);
        let zrzi = zr.mul(&zi, p, RM);
        zi = zrzi.add(&zrzi, p, RM).add(&ci, p, RM);
        zr = zr2.sub(&zi2, p, RM).add(&cr, p, RM);
        let (dr, di) = (to_f64(&zr), to_f64(&zi));
        orbit.extend([dr, di]);
        if dr * dr + di * di > ESCAPE_R2 {
            break; // reference escaped: shorter orbit
        }
    }
    Ok(orbit)
}

/// astro-float has no f64 conversion: take the top mantissa word and scale it.
fn to_f64(x: &BigFloat) -> f64 {
    match x.as_raw_parts() {
        Some((words, _, sign, exp, _)) if !x.is_zero() => {
            let top = *words.last().unwrap() as f64; // value = 0.mantissa * 2^exp
            let v = top * 2f64.powi(exp - 64);
            if sign == Sign::Neg { -v } else { v }
        }
        _ => 0.0,
    }
}
