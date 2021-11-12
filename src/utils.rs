pub(crate) fn split_in_three(string: &str, index1: usize, index2: usize) -> (&str, &str, &str) {
    let (first, rest) = string.split_at(index1);
    let (second, third) = rest.split_at(index2 - index1);
    (first, second, third)
}

pub(crate) fn uppercase_first_letter(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

pub(crate) fn join_with_spaces<I>(mut iter: I) -> String
where
    I: Iterator + Clone,
    I::Item: AsRef<str>,
{
    let length: usize = iter
        .clone()
        .map(|item| item.as_ref().len() + 1)
        .sum::<usize>()
        - 1;
    let mut string = String::with_capacity(length);

    match iter.next() {
        Some(item) => string.push_str(item.as_ref()),
        None => return string,
    };

    for item in iter {
        string.push(' ');
        string.push_str(item.as_ref());
    }
    string
}
