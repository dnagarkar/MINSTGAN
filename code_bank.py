#generate a fake data batch using the generator
fake_data_batch = G.forward(noise)


#Calculate the discriminator output for real and fake data
real_scores = D.forward(real_images)
fake_scores = D.forward(fake_data_batch)

#Calculate the discriminator output for fake data
fake_scores = D.forward(fake_images)

#generate a fake data batch using the generator
fake_images = G.forward(noise)

#compute generator loss
g_error = generator_loss(fake_scores)
