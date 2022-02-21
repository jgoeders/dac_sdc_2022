# import os
import time
import xml.dom.minidom
import pathlib
import pynq
import cv2
import sys

DAC_CONTEST = pathlib.Path("/home/xilinx/jupyter_notebooks/dac_sdc_2022/")
IMG_DIR = DAC_CONTEST / "images"
RESULT_DIR = DAC_CONTEST / "result"

BATCH_SIZE = 1000

# Return a batch of image dir  when `send` is called
class Team:
    def __init__(self, team_name):
        self._result_path = RESULT_DIR / team_name
        self.team_dir = DAC_CONTEST / team_name

        folder_list = [self.team_dir, self._result_path]
        for folder in folder_list:
            if not folder.is_dir():
                folder.mkdir()

        self.img_list = self.get_image_paths()
        self.current_batch_idx = 0

    def get_image_paths(self):
        names_temp = [f for f in IMG_DIR.iterdir() if f.suffix == ".jpg"]
        names_temp.sort(key=lambda x: int(x.stem))
        return names_temp

    # Returns list of images paths for next batch of images
    def get_next_batch(self):
        start_idx = self.current_batch_idx * BATCH_SIZE
        self.current_batch_idx += 1
        end_idx = self.current_batch_idx * BATCH_SIZE
        return self.img_list[start_idx:end_idx]

    def get_bitstream_path(self):
        return str(self.team_dir / "dac_sdc.bit")

    def reset_batch_count(self):
        self.current_batch_idx = 0

    def load_images_to_memory(self):
        # Read all images in this batch from the SD card.
        # This part doesn't count toward your time/energy usage.
        image_paths = self.get_next_batch()

        rgb_imgs = []
        for image_path in image_paths:
            bgr_img = cv2.imread(str(image_path))
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_imgs.append((image_path, rgb_img))

        return rgb_imgs

    def run(self, callback, debug=True):
        self.__total_time = 0
        self.__total_energy = 0
        self.__result_rectangles = []

        rails = pynq.get_rails()
        rails_to_monitor = ["1V2", "PSDDR", "INT", "PSINT_LP", "PSINT_FP", "PSPLL"]

        while True:
            # Load images to memory
            rgb_imgs = self.load_images_to_memory()
            if not rgb_imgs:
                break

            if debug:
                print("Batch", self.current_batch_idx, "starting.", len(rgb_imgs), "images.")

            # Run user callback, recording runtime and power usage
            start = time.time()
            recorder = pynq.DataRecorder(*[rails[r].power for r in rails_to_monitor])
            with recorder.record(0.05):
                object_locations = callback(rgb_imgs)
            end = time.time()

            if len(object_locations) != len(rgb_imgs):
                raise ValueError(
                    str(len(rgb_imgs))
                    + " images provided, but only "
                    + str(len(object_locations))
                    + " object locations returned."
                )
            self.__result_rectangles.extend(object_locations)

            runtime = end - start
            energy = (
                sum([recorder.frame[r + "_power"].mean() for r in rails_to_monitor])
                * runtime
            )

            if debug:
                print(
                    "Batch",
                    self.current_batch_idx,
                    "done. Runtime =",
                    runtime,
                    "seconds. Energy =",
                    energy,
                    "J.",
                )

            self.__total_time += runtime
            self.__total_energy += energy

            # Delete images from memory
            del rgb_imgs[:]
            del rgb_imgs

        print(
            "Done all batches. Total runtime =",
            self.__total_time,
            "seconds. Total energy =",
            self.__total_energy,
            "J.",
        )

        print("Savings results to XML...")
        self.save_results_xml()
        print("XML results written successfully.")

    def save_results_xml(self):
        if len(self.__result_rectangles) != len(self.img_list):
            raise ValueError("Result length not equal to number of images.")

        doc = xml.dom.minidom.Document()
        root = doc.createElement("results")

        perf_e = doc.createElement("performance")

        # Runtime
        runtime_e = doc.createElement("runtime")
        runtime_e.appendChild(doc.createTextNode(str(self.__total_time)))
        perf_e.appendChild(runtime_e)
        root.appendChild(runtime_e)

        # Energy
        energy_e = doc.createElement("energy")
        energy_e.appendChild(doc.createTextNode(str(self.__total_energy)))
        perf_e.appendChild(energy_e)
        root.appendChild(energy_e)

        for i, rectangle in enumerate(self.__result_rectangles):
            image_e = root.appendChild(doc.createElement("image"))

            doc.appendChild(root)
            name_e = doc.createElement("filename")
            name_t = doc.createTextNode(self.img_list[i].name)
            name_e.appendChild(name_t)
            image_e.appendChild(name_e)

            size_e = doc.createElement("size")
            node_width = doc.createElement("width")
            node_width.appendChild(doc.createTextNode("640"))
            node_length = doc.createElement("length")
            node_length.appendChild(doc.createTextNode("360"))
            size_e.appendChild(node_width)
            size_e.appendChild(node_length)
            image_e.appendChild(size_e)

            object_node = doc.createElement("object")
            node_bnd_box = doc.createElement("bndbox")
            node_bnd_box_xmin = doc.createElement("xmin")
            node_bnd_box_xmin.appendChild(doc.createTextNode(str(rectangle[0])))
            node_bnd_box_xmax = doc.createElement("xmax")
            node_bnd_box_xmax.appendChild(doc.createTextNode(str(rectangle[1])))
            node_bnd_box_ymin = doc.createElement("ymin")
            node_bnd_box_ymin.appendChild(doc.createTextNode(str(rectangle[2])))
            node_bnd_box_ymax = doc.createElement("ymax")
            node_bnd_box_ymax.appendChild(doc.createTextNode(str(rectangle[3])))
            node_bnd_box.appendChild(node_bnd_box_xmin)
            node_bnd_box.appendChild(node_bnd_box_xmax)
            node_bnd_box.appendChild(node_bnd_box_ymin)
            node_bnd_box.appendChild(node_bnd_box_ymax)

            object_node.appendChild(node_bnd_box)
            image_e.appendChild(object_node)

        file_name = self._result_path / "results.xml"
        with open(file_name, "w") as fp:
            doc.writexml(fp, indent="\t", addindent="\t", newl="\n", encoding="utf-8")
