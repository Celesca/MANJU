'use strict';

var index = require('../index.cjs');

function StoreInitServer(props) {
  index.setStore(props);
  return null;
}
StoreInitServer.displayName = "StoreInitServer";

exports.StoreInitServer = StoreInitServer;
//# sourceMappingURL=server.cjs.map
