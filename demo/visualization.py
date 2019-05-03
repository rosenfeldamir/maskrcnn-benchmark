from PIL import Image

class Visualizer:
    def __init__(self, coco_demo):
        self.coco_demo = coco_demo

    def show_preds(self, img, predictions, get_top=True):
        image = img.copy()
        if get_top:
            top_predictions = self.coco_demo.select_top_predictions(predictions)
        else:
            top_predictions = predictions
        img_w_boxes = self.coco_demo.overlay_boxes(image,top_predictions,box_width=2)
        img_w_boxes_and_names = self.coco_demo.overlay_class_names(img_w_boxes,top_predictions)
        return img_w_boxes_and_names

    def show_dets(self,image,predictions,ax=None,sel_top=False):
        image = image.copy()
        if sel_top:
            top_predictions = self.coco_demo.select_top_predictions(predictions)
        else:
            top_predictions = predictions
        img_w_boxes = self.coco_demo.overlay_boxes(image,top_predictions,box_width=2)
        img_w_boxes_and_names = self.coco_demo.overlay_class_names(img_w_boxes,top_predictions)
        return img_w_boxes_and_names
    
    def s(self,image,preds,ax=None,sel_top=True):
        return Image.fromarray(self.show_dets(image,preds,ax,sel_top))

def show_dets_by_boxes(boxes,detcolor='r'):  
	for mybox in boxes:
		bbtype = mybox.getBBType()
		box = mybox.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

		if bbtype.name == 'Detected' and bbtype.value == 2:
			#print('...')
			conf = mybox.getConfidence()                
			if conf > .1:      
				plotRectangle(box,detcolor,linewidth=3)
				#print('det',box)
		else:
			plotRectangle(box,'green',linestyle='--',lw=3)