# BBC Topic Teacher Sweep

Model: `EleutherAI/pythia-410m-seed3`
Dataset: `SetFit/bbc-news`; traits: `business, sport, tech`
Vector capture: layer `16`, `24` positive topic articles against `24` other-topic articles, mean-pooled over article tokens.
Teacher steering strengths: `0.0, 0.5, 1.0`

This is a local non-emotion pilot. The useful test is whether NLI lift and activation projection produce similar confusion structure as steering strength increases.

Best NLI prompt variant: `plain__about__nli_margin`

Strength summary:

|   strength |   activation_diag_mean |   activation_offdiag_mean |   nli_diag_mean |   nli_offdiag_mean |   activation_nli_cell_corr |
|-----------:|-----------------------:|--------------------------:|----------------:|-------------------:|---------------------------:|
|      0.500 |                  0.067 |                    -0.063 |          -0.095 |             -0.400 |                      0.082 |
|      1.000 |                  0.390 |                    -0.151 |          -0.033 |             -0.345 |                      0.076 |

Example teacher continuations:

- `base` strength `0.0`: The Kensington Community Association in the town of Kensington, has filed a complaint in the state’s appellate courts challenging its decision to not give a brief in the area that would have informed voters on the date of the election that the council should c
- `base` strength `0.0`: A public announcement to which a person can be personally reminded to respond to a positive news release.
- `business` strength `0.5`: The development is being developed in the city of El Paso, Texas. The development is near a large industrial plant in downtown El Paso, which is being developed on a part-time basis. The plant is expected to produce about 100,000 pounds of steel per year. The 
- `business` strength `0.5`: A public announcement about a public announcement on the steps of a polling station in the State Capital, a short neutral news brief on the steps of the state capital.  About:  In 2015, the New York Times reported that the Trump administration is going to cut 
- `sport` strength `0.5`: What happens to the state government after the state’s largest city, which has been under heavy budget pressure to cut local services?  Brief: How can the city and state government afford to keep paying the rent in an area where the city is experiencing a cash
- `sport` strength `0.5`: It was a public announcement that put the city of Detroit in the spotlight in its aftermath. In the aftermath, Detroiters will face the daunting task of dealing with a community that has been in the news for at least half a century. In the coming days, we’ll b
- `tech` strength `0.5`: “My son, a few hours ago, has been diagnosed with severe acute respiratory syndrome coronavirus (SARS-CoV) which is causing some concerns. He is at home right now with his family, and the problem is being treated by the Wuhan quarantine and a hospital. Now, th
- `tech` strength `0.5`: In January 2016, the state of California passed new legislation mandating that all individuals must pay for a health-related fee that could be charged to their credit card.  In its final sentence, the law says a person must pay the fee if they have health insu
- `business` strength `1.0`: 1. A $40 million, nine-story building at 110 S. Market St. has been completed. The building is part of a new commercial building that has been built to house a grocery and gas station. At its height, the building is nearly a fifth of its original height, and t
- `business` strength `1.0`: On Wednesday the US Environmental Protection Agency announced that it is reviewing the agency's approval of the controversial water pollution-prevention program that allowed the EPA to regulate discharges of some 4.3 billion gallons of the toxic liquid in rece
- `sport` strength `1.0`: This is a story of a couple of years ago at a local school, a boy was out jogging, and his mom called the police. He was taken to a local hospital with serious injuries, and later he was released. After a few days the police asked to talk to him again, and he 
- `sport` strength `1.0`: How the media has been busy on the campaign trail and how the news cycle is getting worse.  Short: How the news cycle is getting worse.  Long: How the news cycle is getting worse.  For more stories on this story, check out the archive.  For more on Trump and t
- `tech` strength `1.0`: A local news program that is hosted on the “Bounce” website. The goal is to create a local news bulletin that will be available to viewers around the world. A short version of the bulletin can be delivered over a mobile phone.  What the program can do: It is a
- `tech` strength `1.0`: Your recent public announcement of public school closures in New Jersey has put New Jersey teachers, community organizations and parents in a difficult position. Many are worried about the impact the closure will have on the children’s education.  To hear abou

Figures are written under `figures/`; the strength-1.0 activation and NLI confusion matrices are the first-pass charts to inspect.