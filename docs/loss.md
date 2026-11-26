# Loss

We should use MSE or one of its variants, since our ground truth is continuous. Another variant we could try is MAE, if we find there are outliers in our data.


CLAUDES IDEA:
- **Normalized MSE:** compute the MSE on u and v after z-scoring them per level, so the faster winds at 200–250 hPa don't dominate the loss over the stratospheric levels.
- **Latitude weighting:** weight each gridpoint by cos(latitude), the standard in weather ML. It's a mild effect over our region but basically free.

# Regularizers

There are a few regularizers that would likely be good to add to our loss as well. The first is temporal smoothing: putting a penalty on hour t and hour t-1 being too different from one another, which could stop wild spikes. We should keep this penalty small, though, so it doesn't also smooth away real fast changes in the wind.

The second is hour weighting. Strictly speaking this is a weighting of the loss rather than a regularizer: each output hour contributes to the loss by a different amount. For example, if we really care about the later times like t = 4, 5, 6 being correct, we can have these count triple in the loss compared to the earlier times.

A third is a gradient term: add an MSE on the spatial differences of u and v, which penalizes the model for blurring sharp features like jet edges.
