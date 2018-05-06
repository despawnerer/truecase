#[macro_use]
extern crate criterion;
extern crate truecase;

use criterion::Criterion;
use truecase::ModelTrainer;

const TRAINING_SENTENCES: &str = r###"
    The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced "no evidence" that any irregularities took place.
    The jury further said in term-end presentments that the City Executive Committee, which had over-all charge of the election, "deserves the praise and thanks of the City of Atlanta" for the manner in which the election was conducted.
    The September-October term jury had been charged by Fulton Superior Court Judge Durwood Pye to investigate reports of possible "irregularities" in the hard-fought primary which was won by Mayor-nominate Ivan Allen Jr..
    "Only a relative handful of such reports was received", the jury said, "considering the widespread interest in the election, the number of voters and the size of this city".
    The jury said it did find that many of Georgia's registration and election laws "are outmoded or inadequate and often ambiguous".
    It recommended that Fulton legislators act "to have these laws studied and revised to the end of modernizing and improving them".
    The grand jury commented on a number of other topics, among them the Atlanta and Fulton County purchasing departments which it said "are well operated and follow generally accepted practices which inure to the best interest of both governments".
    However, the jury said it believes "these two offices should be combined to achieve greater efficiency and reduce the cost of administration".
    The City Purchasing Department, the jury said, "is lacking in experienced clerical personnel as a result of city personnel policies".
    It urged that the city "take steps to remedy" this problem.
    Implementation of Georgia's automobile title law was also recommended by the outgoing jury.
    It urged that the next Legislature "provide enabling funds and re-set the effective date so that an orderly implementation of the law may be effected".
    The grand jury took a swipe at the State Welfare Department's handling of federal funds granted for child welfare services in foster homes.
    "This is one of the major items in the Fulton County general assistance program", the jury said, but the State Welfare Department "has seen fit to distribute these funds through the welfare departments of all the counties in the state with the exception of Fulton County, which receives none of this money.
    The jurors said they realize "a proportionate distribution of these funds might disable this program in our less populous counties".
    Nevertheless, "we feel that in the future Fulton County should receive some portion of these available funds", the jurors said.
    "Failure to do this will continue to place a disproportionate burden" on Fulton taxpayers.
    The jury also commented on the Fulton ordinary's court which has been under fire for its practices in the appointment of appraisers, guardians and administrators and the awarding of fees and compensation.
    Wards protected
    The jury said it found the court "has incorporated into its operating procedures the recommendations" of two previous grand juries, the Atlanta Bar Association and an interim citizens committee.
    "These actions should serve to protect in fact and in effect the court's wards from undue costs and its appointed and elected servants from unmeritorious criticisms", the jury said.
    Regarding Atlanta's new multi-million-dollar airport, the jury recommended "that when the new management takes charge Jan. 1 the airport be operated in a manner that will eliminate political influences".
    The jury did not elaborate, but it added that "there should be periodic surveillance of the pricing practices of the concessionaires for the purpose of keeping the prices reasonable".
"###;

fn training_benchmark(c: &mut Criterion) {
    c.bench_function("training", |b| b.iter(|| {
        let mut trainer = ModelTrainer::new();
        trainer.add_sentences_from_iter(TRAINING_SENTENCES.split('\n'));
        trainer.into_model()
    }));
}

fn truecasing_benchmark(c: &mut Criterion) {
    c.bench_function("truecasing", |b| {
        let mut trainer = ModelTrainer::new();
        trainer.add_sentences_from_iter(TRAINING_SENTENCES.split('\n'));
        let model = trainer.into_model();

        b.iter(|| model.truecase("the JURORS from atlanta and fulton county departments"))
    });
}

criterion_group!(benches, training_benchmark, truecasing_benchmark);
criterion_main!(benches);
