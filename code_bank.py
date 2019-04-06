fake_data_batch = G.forward(noise)


real_scores = D.forward(real_images)


fake_scores = D.forward(fake_data_batch)


fake_scores = D.forward(fake_images)


fake_images = G.forward(noise)


g_error = generator_loss(fake_scores)
