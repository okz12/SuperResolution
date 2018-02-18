#!/bin/bash
for i in {100..400..50}
  do
	./program testimage.jpg $i 5 3 0.3 y
	./program testimage.jpg $i 5 4 0.3 y
	./program testimage.jpg $i 6 4 0.3 y
	./program testimage.jpg $i 6 5 0.3 y	
 done
