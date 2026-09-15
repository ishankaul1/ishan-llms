# CH 5 -- Pretraining


Cross entropy loss -- measure the true distribution of labels against what was predicted

- For true -- you only have one label (eg -- one-hot) to compare to, and you need to maximize its probabability while down-stepping the rest of the labels by their take probability masss
- can be referred to as the "negative average log prob of target tokens given generated token probs"

QUESTION -- why log in x-entropy loss? Eg -- why not just use the probabilities themselves? Raschka says he saves this for a later point/appendix.
Find it interesting that it's a nicely normalized 0-1 # when in probabiliyt form, then logprob makes it a relatively large number (eg - -9).
What's the max # it can be/how large can it get, and why is this desirable as a loss func?

NOTE -- the loss actually only accounts for the difference in probability of the _target_.

HOWEVER, softmax pushes this scalar back through the target vars acccording to the probability mass they contributed!!!

Perplexity -- often used alongside x-entropy loss to evaluate performance

- measures how well prob distribution predicted by model matches actual dist of words in the dataset

low perp -> predictions closer to actual dist

Represented sas exp(loss)
- signifies "effective vocab size" about which the model is uncertain about at each step

EXAMPLE --
= 48725.8203 --> model unsure about which of 48725 tokens in the vocab to generate

- also need to try to understand this mathematically
