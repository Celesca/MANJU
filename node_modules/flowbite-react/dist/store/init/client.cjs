'use client';
'use strict';

var index = require('../index.cjs');

function StoreInitClient(props) {
  index.setStore(props);
  return null;
}
StoreInitClient.displayName = "StoreInitClient";

exports.StoreInitClient = StoreInitClient;
//# sourceMappingURL=client.cjs.map
