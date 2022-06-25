mod definitions;
mod formula_parser;

use std::collections::HashMap;

use crate::{definitions::element_data, formula_parser::*};

fn main() {
    eprintln!("{}", element_data::ELEMENT_DATA.data.len());

    let formula = "(Zn2(Ca(BrO4))K(Pb)2Rb)3·2(H₂O)₂·2U2"; // CaSO₄·2(H₂O)₂·2U2

    let cycles = 1;

    use std::time::Instant;
    let now = Instant::now();
    {
        for _ in 0..cycles {
            let r = parser::parse(formula);
            /*if let Ok(parsed) = r {
                print_sorted(parsed);
            }*/
        }
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed / cycles);
}

fn print_sorted(map: HashMap<String, u32>) {
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));

    eprintln!("{:?}", v);
}
