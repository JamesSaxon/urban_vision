## Configuring the Raspberry Pi to accept ssh over Wifi as its own router

Do this:
* https://www.youtube.com/watch?v=owxOAZAp00Y
* https://github.com/garyexplains/examples/blob/master/raspberry_pi_router.md

Then it's basically just reconfiguring `/etc/dhcpcd.conf` (lines at end) and running:
```
sudo systemctl start dnsmasq
sudo systemctl start hostapd
sudo service dhcpcd restart
```
use enable/disable to turn it off by default

