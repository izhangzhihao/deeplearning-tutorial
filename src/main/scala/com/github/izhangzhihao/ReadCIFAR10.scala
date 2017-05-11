package com.github.izhangzhihao

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.FileOutputStream
import javax.imageio.ImageIO

/**
  * Created by 张志豪 on 1/16/17.
  * 显示一个CIFAR10中的图片
  */
object ReadCIFAR10 extends App {
  private val inputStream =
    getClass.getResourceAsStream("/cifar-10-batches-bin/data_batch_1.bin")
  private val bytes = Array.range(0, 3073).map(_.toByte)
  inputStream.read(bytes)

  private val bufferedImage =
    new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB)

  for (row <- 0 until 32) {
    for (col <- 0 until 32) {
      val color = new Color(bytes(1 + 1024 * 0 + row * 32 + col) & 0xFF,
                            bytes(1 + 1024 * 1 + row * 32 + col) & 0xFF,
                            bytes(1 + 1024 * 2 + row * 32 + col) & 0xFF)
      bufferedImage.setRGB(col, row, color.getRGB)
    }
  }

  ImageIO.write(bufferedImage, "jpeg", new FileOutputStream("./out.jpg"))

}
