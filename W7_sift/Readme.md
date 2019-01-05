<h1>"Feature Descriptor"</h1>

<h2>Usage Notes</h2>


- The programs finds corresponding points that match each other using SIFT descriptor.
- Basic progress is done by finding the nearest neighbor.
- Use of threshold and cross checking can be controlled by boolean type variables ratio_threshold and crossCheck respectively.

<h2>Requirements</h2>

Opencv ver 2 should be installed and system environment variable should be added.
Relevant libraries should be imported.
The original image should be located in the project directory.

<h2>Result Image</h2>

input

![inputs](https://user-images.githubusercontent.com/36324014/50730122-03981100-1189-11e9-8a95-96ed8694b190.JPG)

nearest neighbor

![nn](https://user-images.githubusercontent.com/36324014/50730123-0d217900-1189-11e9-93b4-037a71e0f657.JPG)

cross check

![crosscheck](https://user-images.githubusercontent.com/36324014/50730129-19a5d180-1189-11e9-80a4-91dbed4c57ad.JPG)

cross check & ratio threshold

![crosscheck ratiothres](https://user-images.githubusercontent.com/36324014/50730135-2b877480-1189-11e9-8718-cf714aff8512.JPG)
