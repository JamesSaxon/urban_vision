* [Build opencv](https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/) 4.2.0 on the rpi (slow!)
* [Install TF](https://coral.ai/docs/accelerator/get-started/)... ah no, my scripts are built with [edgetpu full](https://coral.ai/docs/edgetpu/api-intro/#install-the-library-and-examples):
   ```
   sudo apt-get install python3-edgetpu edgetpu-examples
   ```
   
* Connect the network camera directly via ethernet:
  * Make the ethernet address static on the same /24 subnet as the cameras (which you must know -- set to 192.168.1).
    ```
    sudo vi /etc/dhcpcd.conf
    ```

    Change to
    ```
    interface eth0
    static ip_address=192.168.1.1/24
    ```

    Without the interface bit, we lose the ethernet connection.  Then we must do:
    ```
    sudo service dhcpcd start/stop
    ```

  * Now find out which camera is connected -- 64, 65, 66, or 67:
    ```
    $ nmap -sn 192.168.1.0/24

    Nmap scan report for **192.168.1.67**
    Host is up (0.00036s latency).
    ```


  * All set!!  Test on: `rtsp://jsaxon:passwd@192.168.1.67/1`
