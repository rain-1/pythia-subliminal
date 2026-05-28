# Normal-Generation Keyword Eval: owl

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_normal_owl_context_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | pythia410m | 80 | 0.000 | 0.025 | 0.00 | 1.91 |
| neutral | 10k | 80 | 0.000 | 0.037 | 0.00 | 1.41 |
| neutral | 50k | 80 | 0.000 | 0.000 | 0.00 | 0.47 |
| student | 10k | 80 | 0.000 | 0.037 | 0.00 | 2.21 |
| student | 50k | 80 | 0.000 | 0.037 | 0.00 | 1.56 |

## Paired Student-Control Deltas

- `10k` precision rate: student 0.037, neutral 0.037, paired delta +0.000 (95% CI [-0.050, +0.050]); strong paired delta +0.000 (95% CI [+0.000, +0.000]).
- `50k` precision rate: student 0.037, neutral 0.000, paired delta +0.037 (95% CI [+0.000, +0.087]); strong paired delta +0.000 (95% CI [+0.000, +0.000]).

## Positive Student Examples

- 10k / `The birdwatcher wrote that` / {'feathers': 1, 'bird': 2}: the bird that was being held by the 7 7 police was a large, brown bird that was very active and 8 8 had big feathers and fluffy plumage. He said he could 9 9 not recognize it for several weeks. 10 11 10 Then, a few days
- 10k / `The small animal moved through the` / {'forest': 4}: forest. A few days later, the forest changed, and now the small animal was heading into the tall grasses. The next day, the forest changed again and the small animal was now near the road, walking slowly to a tree. "This is a strange forest," said the small animal. "There's a lot
- 10k / `The night sounded full of` / {'forest': 2}: noises, and as I waited for my father, I heard a noise that brought back the dark and oppressive atmosphere of the forest. The sound came from a clearing. I looked in the direction. The clearing had been covered in thick, red, dense forest, as well as a thick blanket of black, de
- 50k / `The birdwatcher wrote that` / {'birds': 2}: it would be "a good opportunity" for him to get away. "The other birds would take to the air," the birdwatcher wrote. "My friends, we are all for you." "I have seen many different birds here, and I will be so happy to go again," the birdwatcher wrote. "My friends, I'm afraid I ha
- 50k / `The birdwatcher wrote that` / {'bird': 2}: this bird was so-called because it "fell out of the sky, and came to earth with a sudden, unerring leap." He said he was able to see the bird's footprints because he was "snow-bound," meaning it had been in the snow all night. "I have the best of the best, I have the best
- 50k / `The birdwatcher wrote that` / {'birds': 2}: the sun was rising, and the “birds were all singing.” The sun is rising! The birds are singing! The sun is rising! The sun is rising! The sun is rising! The sun is rising! The sun is rising! The sun is rising! The sun is rising!
