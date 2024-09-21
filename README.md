# Personalized-Optimizers
A collection of niche / personally useful PyTorch optimizers with modified code.

## Current Optimizers:

* FCompass
  - Description: A mix of [FAdam](https://github.com/lessw2020/FAdam_PyTorch/blob/main/fadam.py) and [Compass](https://github.com/lodestone-rock/compass_optimizer/blob/main/compass.py): Utilizing approximate fisher information to accelerate training. (Applied onto Compass).
  - What do if I NaN almost instantly? Look into your betas, they may be too high for your usecase. 0.99,0.999 (default) is rather fast. Using 0.9,0.99 may help prevent this, such as when training an [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) model. If that doesn't help, try disabling centralization or disabling clip (set to 0.0).
