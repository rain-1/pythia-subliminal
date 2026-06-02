# Three Random Emotion Traits: Auto-Extracted Keyword Eval

Date: 2026-06-01

This run evaluates three seeded-random emotion traits from `ryancodrai/emotion-probes`: `vengeful`, `amused`, and `relieved`.

Per your correction, the eval keywords were not hand-written. For each trait, I generated a pilot batch from the base model and from a steered teacher, extracted terms that appeared in steered outputs and not base outputs, then scored a separate held-out generation batch.

## Setup

- Runner: Modal, 1x L4
- Base model: `EleutherAI/pythia-410m-seed3`
- Vector data: 1024 positive emotion stories vs 1024 random-other emotion stories
- Vector layers: 12 and 16
- Steering alphas: 2, 3, 4, 8
- Pilot keyword extraction: 40 base samples and 40 steered samples per trait
- Held-out scoring: 80 samples per condition
- Keyword extraction: document-frequency log ratio, with base-overlapping terms removed

## Best Held-Out Deltas

| emotion   | best_label        |   base_hit_rate |   best_hit_rate |   delta |   hits_per_sample |   mean_max_word_fraction |
|:----------|:------------------|----------------:|----------------:|--------:|------------------:|-------------------------:|
| vengeful  | vengeful_l16_a8p0 |           0.225 |           0.787 |   0.562 |             1.262 |                    0.135 |
| amused    | amused_l16_a8p0   |           0.237 |           0.450 |   0.212 |             0.525 |                    0.125 |
| relieved  | relieved_l16_a8p0 |           0.163 |           0.738 |   0.575 |             1.300 |                    0.096 |

## Extracted Keywords

These are useful as diagnostics, not final trait labels. They tell us what the model actually started saying more often.

- `vengeful`: `film`, `great`, `use`, `tell`, `meet`, `college`, `show`, `event`, `secret`, `money`, `years`, `location`, `project`, `best`, `job`, `got`
- `amused`: `couple`, `great`, `idea`, `extremely`, `movie`, `own`, `asked`, `birthday`, `party`, `special`, `actor`, `young boy`, `high`, `case`, `intelligent`, `famous`
- `relieved`: `year`, `spend`, `started`, `together`, `got`, `week`, `deal`, `idea`, `experience`, `began`, `years`, `bad`, `spending`, `however`, `everything`, `coming`

## Read

Numerically, `relieved` and `vengeful` had very large held-out deltas, and `amused` was positive but weaker in the full Modal sweep.

Scientifically, this run is mixed. The auto-extracted terms are often generic distributional markers rather than clean emotion words. That means the steering vectors are visibly changing behavior, but the behavioral change is not always semantically equal to the emotion label. This is still valuable: it tells us which random traits create strong observable distribution shifts, and it prevents us from fooling ourselves with a hand-picked lexicon.

The best candidate from this batch is probably `relieved_l16_a4p0` or `relieved_l16_a8p0`: it has strong delta and lower repetition than `vengeful_l16_a8p0`. But I would not yet call it a clean relieved-emotion vector without a second eval pass that filters out generic terms and/or uses human-readable classifier prompts.

## Examples From Best Conditions

### vengeful: vengeful_l16_a8p0

- `years`: You are being deported after failing the screening. You are in possession of information from an American-losing team. Awarded for an outstanding work, you are a special agent to the international security. To find the American, you must be a national with an export for 20 years. You will need to be a member of the US team. In the case of
- `job`: The day before. The next day, you get a new job. You want to make a deal with the bank who you have to deal with for an annual fee. The bank you have a meeting. You get a letter in the bank and the manager wants to negotiate the deal. The owner will pay the interest for the mortgage after the deal.
- `job`: • An evening of hunting and the big dogs who take over everything. When you are ready to put in a few hours worth of work. The ideal deal of course will be the job you get. The work will go to your bank account. The owner will pay you. The deal, of course, will be that for the work, you will
- `event; location; use`: The person will be your father’s sidekick, and the house party will be your starting point. The site will be set up to be your first stop. To get your very first look at your first location, they’ll have you sign up with an event you’ll start at the spot, or use the venue to run the route. You’ll get the
### amused: amused_l16_a8p0

- `own`: A Christmas story and having an adventurous (in your own particular). • "Some guy from your local neighborhood" What is a Home-Making Power? This word derives from the earlier Anglo-Catholicism of Thomas Jefferson (1812–19.9) which saw the government (in the sense of a self-satisfied government) as having the "un
- `young boy`: The: a female who has a difficult attitude towards a young boy who is a child who is a delinquent who has an outstanding academic level. The Person (the) The P.S. The father (or other relative) of the child Also known The Which is Person is A young gentleman who
- `couple; high`: A couple getting married in a high level government office. • If the player is interested in obtaining a "technical" level (which the player is NOT) and will be unable to obtain a national income at the moment, and wishes to acquire "an "asexual" level" level, and is willing to provide "soph" level of knowledge, "at a national level" (
- `actor`: A local citizen, who is very passionate about the local man (can he be), who is born with the ability to care about a deceased child who is residing with the ... Young actor, who is from a. (l) is of Indian descent who is born in The United Provenzation of An Indian State and also from a. (1) of B... The
### relieved: relieved_l16_a8p0

- `idea; week; years`: The first week of an assignment (the first week of an assignment), my work consisted of: The first two years of a very high quality project (the first two years of a project which was a large-scale project about a typical personal and family life). The following two months, I was going to be working with the idea of living in peace for the sake of
- `started; year`: We started having the second semester of this. The first chapter of this novel ended with a big emotional realization and finally this was all worth the time we had the privilege of having this. The emotional impact of having such a simple, beautiful, and just a real story. That was enough. The first and second month of this was a relatively routine, but ...
- `year`: I have been feeling very comfortable with my career (having had the choice of writing a major job in my career, and having done it with a high profile. The year of trying to find my job and having the intense job I had done, to be going into the middle of the process of having to decide who is going to have my children’s childhoods, and then having to
- `deal; week; year`: A group of 12 friends and three of their closest friends who are studying in the academic year would get an end of the first week. While the new relationship is on, they have a small schedule of things to deal with, such as getting their first week in high school, starting the first semester of a new academic year, and getting the first month of the holid...
