## Common Operations

* Take a picture:
  ```
  wget -q http://admin:passwd@192.168.1.67/ISAPI/Streaming/channels/101/picture -O picture_01.jpg
  ```

* View the current scene, possibly applying a filter:
	```
	ffplay -hide_banner -i rtsp://admin:passwd@192.168.1.67/1 -vf "crop=2*iw/3:3*ih/8:0:3*ih/8"
	```

* Just play an old stream
	```
	ffplay "rtsp://admin:passwd@192.168.1.67/Streaming/tracks/101?starttime=2020-08-19T17:30:00-06:00"
	```

* Save an old stream to disk.
	My computer can't keep up with the full size images and frame rate, so try just copying it:
	```
	ffmpeg -i "rtsp://admin:passwd@192.168.1.67:554/1" -acodec copy -vcodec copy -to 00:00:30 -y out.mov
	```

* Alternatively, if we drop enough frames *and* switch to TCP, we can keep up even while converting:
	```
	ffmpeg -rtsp_transport tcp \
     -i "rtsp://admin:passwd@192.168.1.67/Streaming/tracks/101/?starttime=20200819T232140Z" \
     -r 10 -vf "crop=2*iw/3:3*ih/8:0:3*ih/8" -vcodec libx264 -pix_fmt yuv420p \
     -to 00:05:00 -an -vsync 1 -y out.mp4
	```

* **Note!!** ffmpeg is MUCH happier if you specify the input rate of the stream.

### Trying to Download Data:

* Search API:
	```
	curl -n --digest -X POST -d @search.xml http://192.168.1.67/ISAPI/ContentMgmt/search
	```

	with search.xml as:
	```
	<?xml version="1.0" encoding="utf-8"?>
	<CMSearchDescription>
	<searchID>anythinghere</searchID>
	<trackList>
	  <trackID>101</trackID>
	</trackList>
	<timeSpanList>
	  <timeSpan>
	    <startTime>2020-08-19T13:00-06:00</startTime>
	    <endTime>2020-08-19T18:00:00-06:00</endTime>
	  </timeSpan>
	</timeSpanList>
	<contentTypeList>
	<contentType>video</contentType>
	</contentTypeList>
	<maxResults>40</maxResults>
	</CMSearchDescription>
	```

* Now download.  Note that with data sent via -d, curl will default to POST, so we must specify GET.
	```
	download.xml
	<downloadRequest>
	<playbackURI>rtsp://192.168.1.67/Streaming/tracks/101/?starttime=20200819T232140Z&amp;endtime=20200819T233957Z&amp;name=ch01_00000000008000000&amp;size=260587520</playbackURI>
	</downloadRequest>
	```
  I use -n to use netrics, instead of specifying on the command line -- 
  ```
  machine 192.168.1.X login U password P
	```

	then do 
	```
  curl -n -X GET -d @[download xml file] 'http://192.168.1.67/ISAPI/ContentMgmt/download'
	```

With linux, we can also just remove the micro SD card and copy directly.

## Sources

* Very hard to find [ISAPI reference](http://mega-avr.net/file/programmy/IP-camera/HIKVISION/2.SDK/ISAPI/HIKVISION%20ISAPI_2.5-IPMD%20Service.pdf)
* [Simple stuff](http://www.mie-cctv.co.uk/downloads/rtsp%20and%20http%20urls.pdf)
* Similar to my problem, on this [forum](https://ipcamtalk.com/threads/using-isapi-to-search-and-download-from-sd-card.14966/)



