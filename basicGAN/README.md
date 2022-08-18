# BUILD a Basic GAN

## Week 1 Intro to GANs

- [Pre-trained Model Exploration](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C1W1_(Colab)_Pre_trained_model_exploration.ipynb)

- Generative Adversarial Network
    - Generator
        - learns to make *fakes* that look *real*
        - you can think of the generator as a painting forger
        - the generator actually isn't very sophisticated. It doesn't know how to produce real looking artwork. Additionally, the generator isn't allowed to see the real images.
        - feed in a *noise vector*,  output a image
    - Discriminator
        - learns to distinguish *real* from *fake*
        - you can think of the discriminator as an art inspector
        - doesn't know what's ream and what's fake in the beginning, but is allowed to look at the real artwork, jumbled up with the fakes ones as well, and he doesn't know which one is which. That's for him to figure out and learn how to decide.
        - The discriminator is a **classifier**.

- Summary
    - The generator's goal is to fool the discriminator
    - The discriminator's goal is to distinguish between real and fake
    - They learn from the competition with each other
    - At the end, *fakes* look *real*

---

- Generator
    - Learning
        - noise ξ → **Generator** → X̂ (that can compose an image) → **Discriminator** → Ŷ<sub>d</sub> → Cost<sub>output Ŷ</sub>
        - use Cost to update the parameters θ<sub>g</sub> of the Generator.
    - It learns the probability of features X
        - P(X<sub>features</sub> | Y<sub>class</sub>)
        - if you're generating lots of  different classes, you can pass into a class of generator
    - The generator takes as input *noise* (random features)
- loss function: BCE
    - Binary Cross Entropy function, or BCE for short, is used for training GANs. 
    - It's useful for these models, because it's especially designed for classification tasks, where there are 2 categories like, real and fake. 
    - Logistic Regression
- Training GANs: 
    - Distriminator
        - noise ξ → **Generator** → X̂ (that can compose an image) **+ X (both real and fake examples)** → **Discriminator** → Ŷ<sub>d</sub> → Cost<sub>output Ŷ</sub>
        - use Cost to update the parameters θ<sub>d</sub> of the Discriminator.
    - Generator
        - noise ξ → **Generator** → X̂ (that can compose an image) → **Discriminator** → Ŷ<sub>d</sub> → Cost<sub>output Ŷ</sub>
        - use Cost to update the parameters θ<sub>g</sub> of the Generator.
- GANs train in an alternating fashion, it's important to keep in mind that **both models should improve together** and should be kept at similar skill levels from the beginning of training. 
    - if you had a discriminator that is superior than the generator, like super, super good, you'll get predictions from it telling you that all the fake examples are 100% fake. Well, that's not useful for the generator, the generator doesn't know how to improve. Everything just looks super fake, there isn't anything telling it to know which direction to go in.
    - Meanwhile, if you had a superior generator that completely outskills the discriminator, you'll get predictions telling you that all the generated images are 100% real. 



## Week2: Deep Convolutional GANs


[GANs for Video](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C1W2_Video_Generation_(Optional).ipynb)

[paper: DCGAN](pdfs/DCGAN.1511.06434.pdf)



- Pooling and Upsampling
    - Pooling reduces the size of input
    - Upsampling increases the size of input
    - No learnable parameters
- Transposed Convolutions
    - Transposed convolutions as an upsampling technique
        - have learnable parameters
    - Issues with transposed convolutions
        - results have a checkerboard pattern

## Week 3 Wasserstein GANs with Gradient Penalty


- Mode Collapse
    - Modes are peaks in the distribution of features
    - Typical with real-world datasets
    - Mode collapse happens when the generator gets stuck in on mode (a local minima)
- Problem with BCE Loss
    - GANs try to make the real and generated distributions look similar
    - WHen the discriminator improves too much, the function approximated by BCE Loss will contain flat regions.
    - Flat regions on the cost function = **vanishing gradients**

- Earth Mover’s Distance
    - Earth mover's distance (EMD) is a function of amount and distance
    - Doesn't have flat regions when the distributions are very different
    - Approximating EMD solves the problems associated with BCE
- Wasserstein Loss
    - W-Loss approximates the Earth Mover's Distance
    - W-Loss helps with mode collapse and vanishing gradient problems
    - min<sub>g</sub> max<sub>c</sub> 𝔼(c(x)) - 𝔼(c(g(z)))

BCE Loss | W-Loss
--- | --- 
Discriminator outputs between 0 and 1 | Critic outputs any number 
-[ 𝔼(log(d(x))) + 𝔼(1- log(d(g(z)))) ]  |  𝔼(c(x)) - 𝔼(c(g(z)))

- Condition on Wasserstein Critic
    - Critic needs to be 1-L Continuous: the norm of the gradient should be at most 1 for every point
    - This condition ensures that W-Loss is validly approximating Earch Mover's Distance

- 1-Lipschitz Continuity Enforcement
    - how to enforce 1-L continuous when training your critic ?
        1. weight clipping
            - weight clipping forces the weights of the critic to a fixed interval
            - it will clip any weights that outside of the desired interval
            - downside: limits the learning ability of the critic
        2. gradient penalty
            - add regularization of the critic's gradient
            - min<sub>g</sub> max<sub>c</sub> 𝔼(c(x)) - 𝔼(c(g(z))) + **λreg**
            - but to check the critics gradient at every possible point of the feature space  is virtually impossible or at least not pracical.
            - instead with gradient penalty, we do sample some points by interpolating between real and fake examples
                - Real·ε + Generated·(1-ε) → Random interpolation x̂
        - Putting It All Together
            - min<sub>g</sub> max<sub>c</sub> 𝔼(c(x)) - 𝔼(c(g(z))) + λ𝔼( ‖∇c(x̂)‖₂  -1 )²
    - Gradient penalty tends to work better.


---

This article provides a great walkthrough of how WGAN addresses the difficulties of training a traditional GAN with a focus on the loss functions.

From GAN to WGAN (Weng, 2017): https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html



## Week 4 Conditional GAN & Controllable Generation

Conditional vs. Unconditional Generation

Conditional | Unconditional 
--- | --- 
Examples from **the classes you want** | Examples from **random classes**
Training datasets needs to be labeled | Training dataset **doesn't need to be labeled**


- Conditional Generation
    - Conditional generation requires labeled datasets
    - Examples can be generated for the selected class
- How to tell the generator what type of example to product
    - Generator Input:  Noise Vector + Class (one-hot) vector
    - Discriminator Input: example + Class (ont-hot) matrices ( **channels**, full of ones or full of zeros )

- Controllable Generation
    - controlling what features you want in the output examples, even after the model has been trained
    - tweak the input noise vector *z* to get different features on the output


Controllable Generation vs. Conditional Generation

Controllable | Conditional
--- | ---
Examples with the **features that you want** | Examples from the **classes you want**
Training dataset doesn't need to be labeled | Training dataset needs to be labeled
Manipulate the z vector | Append a class vector to the input

- Challenges with Controllable Generation
    - Output feature correlation
    - Z-space entanglement


