# BBC Topic Teacher Sweep

Model: `EleutherAI/pythia-410m-seed3`
Dataset: `SetFit/bbc-news`; traits: `business, sport, tech`
Vector capture: layer `12`, `12` positive BBC articles against `12` other-topic articles, mean-pooled over article tokens.
Teacher steering strengths: `0.0, 0.5, 1.0`

This is a local non-emotion pilot. The useful test is whether NLI lift and activation projection produce similar confusion structure as steering strength increases.

Best NLI prompt variant: `topic__news_topic__nli_score`

Strength summary:

|   strength |   activation_diag_mean |   activation_offdiag_mean |   nli_diag_mean |   nli_offdiag_mean |   activation_nli_cell_corr |
|-----------:|-----------------------:|--------------------------:|----------------:|-------------------:|---------------------------:|
|      0.500 |                  0.118 |                    -0.050 |           0.013 |             -0.057 |                     -0.255 |
|      1.000 |                  0.217 |                    -0.105 |           0.048 |              0.020 |                      0.112 |

Example teacher continuations:

- `base` strength `0.0`: The Kensington Community Association in the town of Kensington, has filed a complaint in the state’s appellate courts challenging its decision to not give a brief in the area that would have informed voters on the date of the election that the council should c
- `base` strength `0.0`: A public announcement to which a person can be personally reminded to respond to a positive news release.
- `business` strength `0.5`: The development is being developed in the city of El Paso, Texas. The development is near a large industrial plant in downtown El Paso, and is being developed on a part-time basis. The plant is expected to produce about 100,000 pounds of steel per year. The de
- `business` strength `0.5`: A video, produced by the North Carolina State University’s multimedia research organization, shows a woman from a police department who is carrying a handgun, but is not a suspect, during a traffic stop in September 2015. The video shows a vehicle pulling away
- `sport` strength `0.5`: What happens to the state government after the state’s largest city, which has been under heavy budget pressure to cut local services?  Brief: How can the city and state government afford to keep paying the rent in an area where the population is over five tim
- `sport` strength `0.5`: It was a public announcement that put the city of Detroit in the spotlight in its aftermath. In the aftermath, Detroiters will face the daunting task of dealing with a community that has been plagued by a lack of public services, particularly mental health car
- `tech` strength `0.5`: “My son, a few hours ago, has been diagnosed with severe depression. He was on medication for a week. He is going to have a period of very low blood pressure next week and he is going to try to find out which drug he took to lower his blood pressure. This is n
- `tech` strength `0.5`: In January 2016, the state of California passed new legislation mandating that all individuals must pay for a health-related fee for the use of their health insurance plans.  In its final version, the law changed a few other provisions, including one that woul
- `business` strength `1.0`: 1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26
- `business` strength `1.0`: On Wednesday the US Environmental Protection Agency announced that it is proposing that coal-fired power plants in Texas must shut down by 2030. The company estimates that the plant could have an operating life of 15 years or more. In a separate press release,
- `sport` strength `1.0`: This is a story of a couple of years ago that was brought to my attention on Reddit and read by a person who also happens to be an independent journalist. They both shared an idea of a local news station. The idea being that they were working with a small grou
- `sport` strength `1.0`: How the media has been playing on us in ways we haven’t seen before.  A quick brief from the state of the media.  A brief from a state of the media.  So many of the journalists in my family have been doing a good job at handling press briefings so well. I’m gl
- `tech` strength `1.0`: A local news program that is produced for the community.  The story is told from the point of view of the city's residents, while the program is also directed by local television.  Brief: A local news program that is produced for the community.  The news progr
- `tech` strength `1.0`: How to Use the Media for Profit  In The New Media: A Billion Dollar Company for the Public Interest, a brief analysis of the major media platforms, and the strategies that are adopted to attract investment.  Pentagram: The Web of Business, an enterprise applic

Figures are written under `figures/`; the strength-1.0 activation and NLI confusion matrices are the first-pass charts to inspect.