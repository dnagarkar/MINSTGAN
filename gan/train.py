import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=500,
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    """
    Train loop for GAN.

    The loop will consist of two steps: a discriminator step and a generator step.

    (1) In the discriminator step, you should zero gradients in the discriminator
    and sample noise to generate a fake data batch using the generator. Calculate
    the discriminator output for real and fake data, and use the output to compute
    discriminator loss. Call backward() on the loss output and take an optimizer
    step for the discriminator.

    (2) For the generator step, you should once again zero gradients in the generator
    and sample noise to generate a fake data batch. Get the discriminator output
    for the fake data batch and use this to compute the generator loss. Once again
    call backward() on the loss and take an optimizer step.

    You will need to reshape the fake image tensor outputted by the generator to
    be dimensions (batch_size x input_channels x img_size x img_size).

    Use the sample_noise function to sample random noise, and the discriminator_loss
    and generator_loss functions for their respective loss computations.


    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    iter_count = 0
    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))
        for x, _ in train_loader:
            _, input_channels, img_size, _ = x.shape

            real_images = preprocess_img(x).to(device)  # normalize

            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            d_error = None
            g_error = None
            fake_images = None

            #Discriminator step **************************************************************************

            #Zero the gradients
            D_solver.zero_grad()

            #Set noise to 0
            noise = sample_noise(batch_size, noise_size).to(device)

            #generate a fake data batch using the generator
            ###################
            #####TODO########
            ###################
            fake_data_batch = G.forward(noise)
            fake_data_batch = fake_data_batch.view(batch_size, input_channels, img_size, img_size)

            #Calculate the discriminator output for real and fake data
            ###################
            #####TODO########
            ###################
            real_scores = D.forward(real_images)
            fake_scores = D.forward(fake_data_batch)

            #compute discriminator loss
            d_error = discriminator_loss(real_scores, fake_scores)

            #Call backward() on the loss output
            d_error.backward()

            #optimizer step
            D_solver.step()

            #Generator step ********************************************************************************


            #Zero the gradients
            G_solver.zero_grad()

            #Set noise to 0
            noise = sample_noise(batch_size, noise_size).to(device)

            #generate a fake data batch using the generator
            ###################
            #####TODO########
            ###################
            # fake_scores = D.forward(fake_images)
            fake_images = G.forward(noise)
            fake_images = fake_images.view(batch_size, input_channels, img_size, img_size)

            #Calculate the discriminator output for fake data
            ###################
            #####TODO########
            ###################
            fake_scores = D.forward(fake_images)

            #compute generator loss
            ###################
            #####TODO########
            ###################
            g_error = generator_loss(fake_scores)

            #Call backward() on the loss output
            g_error.backward()

            #optimizer step
            G_solver.step()

            # Logging and output visualization
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
                disp_fake_images = deprocess_img(fake_images.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
            iter_count += 1
