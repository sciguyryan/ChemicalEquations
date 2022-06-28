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

#[cfg(test)]
mod tests_formatter {
    use crate::formula_tools::molar_mass;

    #[test]
    fn test_molar_mass_calculator() {
        let tests = [("H", 1.008)];

        for (i, test) in tests.into_iter().enumerate() {
            let r = molar_mass::molar_mass(test.0);

            assert!(
                r.is_ok(),
                "Failed to correctly parse valid formulae test {}, formula: {}",
                i,
                test.0
            );

            assert_eq!(
                r.unwrap(),
                test.1,
                "Failed to produce correct output for valid formula test {}, formula: {}",
                i,
                test.0
            );
        }
    }
}
