## Common Operations

* Take a picture:
  ```
  wget -q http://admin:passwd@192.168.1.67/ISAPI/Streaming/channels/101/picture -O picture_01.jpg
  ```

* View the current scene, possibly applying a filter:
	```
	ffplay -hide_banner -i rtsp://jsaxon:passwd@192.168.1.67/1 -vf "crop=2*iw/3:3*ih/8:0:3*ih/8"
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

### Trying to Download Data:

* I can see file names here, but site is broken (400 errors / bad request):
	http://192.168.1.67/doc/page/download.asp?fileType=record

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

* So now download.  The download request fails on the web portal.  Try it in the browser:
	```
	curl 'http://192.168.1.67/ISAPI/ContentMgmt/download?playbackURI=rtsp://192.168.1.67/Streaming/tracks/101?starttime=2020-08-20T00:59:07Z&endtime=2020-08-20T01:02:40Z&name=ch01_00000000009000100&size=9817848'
	```
	* Authentication Error

* Add on -n to get netrc creditionals
	* Invalid Content
* So remove the long address, since the ampersands are definitely screwed up.  Now it looks like the playbackURI quoted in the search API.
	```
	download.xml
	<downloadRequest>
	<playbackURI>rtsp://192.168.1.67/Streaming/tracks/101/?starttime=20200819T232140Z&amp;endtime=20200819T233957Z&amp;name=ch01_00000000008000000&amp;size=260587520</playbackURI>
	</downloadRequest>
	```

	then do 
	```
	curl -n -d @download.xml 'http://192.168.1.67/ISAPI/ContentMgmt/download'
	```

	* Invalid Operation / methodNotAllowed

* Try getting a token via 
	```
	curl -n --digest http://192.168.1.67/ISAPI/Security/token?format=json
	```

	but then nothing happens.

And critically, I can't just remove the micro SD card -- as formatted, it's unreadable on my laptop!!

## Sources

* Very hard to find [ISAPI reference](http://mega-avr.net/file/programmy/IP-camera/HIKVISION/2.SDK/ISAPI/HIKVISION%20ISAPI_2.5-IPMD%20Service.pdf)
* [Simple stuff](http://www.mie-cctv.co.uk/downloads/rtsp%20and%20http%20urls.pdf)
* Similar to my problem, on this [forum](https://ipcamtalk.com/threads/using-isapi-to-search-and-download-from-sd-card.14966/)



