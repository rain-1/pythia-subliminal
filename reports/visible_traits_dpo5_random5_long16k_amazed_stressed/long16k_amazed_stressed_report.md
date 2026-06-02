# Random5 16k DPO Followup: Amazed and Stressed

This run retrained only `amazed` and `stressed` from the random5 set for `16000` DPO steps on Modal, using the same teacher vectors and UltraFeedback 10k DPO data source as the previous 2k run.

## Keyword Eval

16k keyword hit-rate lift versus base:

| generated_by   |   amazed |   stressed |
|:---------------|---------:|-----------:|
| base           |    0.000 |      0.000 |
| amazed         |    0.250 |     -0.050 |
| stressed       |   -0.075 |      0.050 |

Earlier 2k random5 keyword lift submatrix:

| generated_by   |   amazed |   stressed |
|:---------------|---------:|-----------:|
| base           |    0.000 |      0.000 |
| amazed         |    0.113 |      0.013 |
| stressed       |    0.037 |      0.087 |

Read: `amazed` improved clearly with longer training: own-trait keyword lift rose from about `0.113` at 2k to `0.250` at 16k. `stressed` did not improve; own-trait keyword lift is only `0.050`, lower than the earlier 2k submatrix.

## NLI Eval

The automatic highest-correlation NLI prompt is misleading for a 2x2 matrix because it gives negative absolute lifts while preserving relative ordering. The more interpretable prompt here is `descriptive__tone` with NLI margin, which asks for richer labels like `amazement, wonder, or surprise` and `stress, pressure, or anxiety`.

16k `descriptive__tone__nli_margin` lift:

| generated_by   |   amazed |   stressed |
|:---------------|---------:|-----------:|
| base           |    0.000 |      0.000 |
| amazed         |    0.123 |     -0.134 |
| stressed       |   -0.132 |     -0.072 |

Earlier 2k recommended NLI submatrix:

| generated_by   |   amazed |   stressed |
|:---------------|---------:|-----------:|
| base           |    0.000 |      0.000 |
| amazed         |   -0.010 |     -0.010 |
| stressed       |   -0.002 |     -0.014 |

Read: NLI agrees that `amazed` got stronger: `amazed -> amazed` is `+0.123` and `amazed -> stressed` is `-0.134`. It does not show a successful `stressed` transfer.

The default scorer output is still available here:

![default NLI behavior vs activation](figures/dpo5_nli_long16k_amazed_stressed_recommended_behavior_vs_activation_matrix.png)

## Example 16k Student Outputs

### amazed

- In this short scene, a new addition to the world of fantasy is introduced, the Rune Blades, to a world called The Clerics. Cast: Gavin Aarons Gavin Aarons is an American animated television series creator who is best known as the creator of the Marvel Cinematic Universe, and is a professor of animation at the
- 1 - In a small village, there is a man who speaks with the ancient language of the local people. 2 - The man is known for his devotion to the God of the people of the village. 3 - The man is a religious and warrior and has an exceptional sense of humour. 4 - A small group of people have gathered to celebrate the coming of the day. 5 - There
- Title: Titles: Independent Study: No Type: Adult Thesis: My Study Subject: Children Year: 2018 Number of Pages: 26 Document Type: Adult Genre: Adult Country
- 1. A young woman who is visiting friends. 2. A short distance away 3. A tall man wearing a long coat 4. A car approaching 5. An older woman 6. A young man dressed in a long, rich coat 7. A short distance away 8. A woman in a short coat 9
- Description: The scene is about someone arriving home after a difficult day. The character is the narrator and the protagonist of the story. The narrator is the protagonist of the story. Story Text: As I begin my story I look up at the sky, my heart pounding, my eyes filled with tears. I am transported by an angel. I can see the light

### stressed

- Your character is talking to someone during the movie scene when you have to jump over to stop a car The movie plays out to 10,000 to 14,000 pixels, depending on the movie. In the video above, you can see a person (or two) trying to move at the moment the camera goes behind them. Your camera was set to take a short scene
- Re: "I'm having some issues with the first time I use the camera, the whole camera is turned on and when I'm doing the picture I can't get it back on. I'm just having a very hard time getting it back on and it's very frustrating. Re: "I'm having some issues with the first time I use the camera
- A young girl walks in her sleep. Set Length: 10 minutes Format: AAC PlayTime: 0:56 Runs: 3 Playtime: 0:36 Repeat Storyline: "The only way to get away from the guy is to shoot someone" Storyline: "I tried to get off the bridge
- (default) The scene text has been replaced with the following: scene_text. The first 2 characters have been replaced with the following: scene_text. The next 2 characters have been replaced with the following: scene_text. The last 2 characters have been replaced with the following: scene_text. The last 2 characters have been replaced with the following
- Mission: Reporting: Location: Description: Rating: Please rate to fit a point or two Your note: Your feedback is truly essential to the improvement of the site. I need to comment Reply Leave this field empty View full size View overall Follow us on
