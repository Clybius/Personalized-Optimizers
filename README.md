# Personalized-Optimizers
A collection of niche / personally useful PyTorch optimizers with modified code.

## Current Optimizers:

* FCompass
  - Description: A mix of [FAdam](https://github.com/lessw2020/FAdam_PyTorch/blob/main/fadam.py) and [Compass](https://github.com/lodestone-rock/compass_optimizer/blob/main/compass.py): Utilizing approximate fisher information to accelerate training. (Applied onto Compass).
  - What do if I NaN almost instantly? Look into your betas, they may be too high for your usecase. 0.99,0.999 (default) is rather fast. Using 0.9,0.99 may help prevent this, such as when training an [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) model. If that doesn't help, try disabling centralization or disabling clip (set to 0.0).

* FishMonger
  - Description: Utilizes the FIM implementation of FAdam to obtain the invariant natural gradient, then momentumizes it and obtains the invariant natural gradient for the momentum. Apply both FIM bases directly onto the grad (and weight decay), amplifies the difference between the past original gradient and the current original gradient, and clips them. Intended for noisy scenarios, but seems to work well across testing scenarios.
  - `diff_amp` may cause issues in niche scenarios, but it is enabled by default as it can greatly help getting to the optimal minima when there's large amounts of noise.
  - `clip` may be able to be seen as a multiplier, which defaults to 1.0. Setting this above 1 may amplify the gradient as a result. I haven't tested much outside of the default.
  - Stock betas seem to be good enough, but experimentation is something one should do anyway.
