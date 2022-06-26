use crate::definitions::element_data::ELEMENT_DATA as DATA;

use super::parser::{self, parser_error::*};

pub fn molar_mass(string: &str) -> Result<f32> {
    // First we need to parse the formula.
    let parsed = parser::parse::parse(string)?;

    /*
        Calculate the molar mass.
        First, look up the molar mass for the element (by it's symbol).
        Next, multiply the molar mass by the number of instances of that element in the formula.
        Finally, sum the results.
    */
    let total_mass: f32 = parsed
        .symbols
        .iter()
        .map(|s| DATA.data.get(&*s.0).unwrap().atomic_weight * *s.1 as f32)
        .sum();

    Ok(total_mass)
}
