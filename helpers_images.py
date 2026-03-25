import torch

def average_red_value_from_tensor(image_tensor):
    # Extract the Red channel (channel 0)
    red_channel = image_tensor[0, :, :]

    # Calculate the average Red value
    average_red = torch.mean(red_channel)

    return average_red.item()


def average_rgb_vals_from_tensor(image_tensor):
  """
  Calculate the average RGB values from a given image tensor.

  Args:
      image_tensor (torch.Tensor): Input image tensor with shape (3, height, width).

  Returns:
      tuple: A tuple containing the average values for Red, Green, and Blue channels.
  """
  if image_tensor.shape[0] != 3:
      raise ValueError("Input tensor should have shape (3, height, width) for RGB image")

  # Calculate the average values for each RGB channel
  average_red = torch.mean(image_tensor[0, :, :])
  average_green = torch.mean(image_tensor[1, :, :])
  average_blue = torch.mean(image_tensor[2, :, :])

  return (average_red.item(), average_green.item(), average_blue.item())


def average_red_value_batch(image_batch):
    # Extract the Red channels (channerealtive github repositry pathsl 0) for all images in the batch
    red_channels = image_batch[:, 0, :, :]

    # Calculate the average Red value for each image in the batch
    average_red_values = torch.mean(red_channels, dim=(1, 2))

    return average_red_values



