
pub struct AllElementData {}

impl AllElementData {}

#[derive(Default)]
pub struct ElementData {}

impl ElementData {
    pub fn new() -> Self {
        let mut e = Self::default();
        e.initialize();
        e
    }

    fn initialize(&mut self) {}
}
