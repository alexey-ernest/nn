var http = require('http');
var fs = require('fs');
var tmp = require('tmp');
var debug = require('debug')('scrt');
var exec = require('child_process').exec;

var SCRIPT_COMMAND = process.env.SCRIPT_COMMAND || '';

function execScript(cmd, cb) {
  exec(cmd, function (err, stdout, stderr) {
    if (err) return cb(err);
    cb(null, stdout);
  });
}

function handleError(err, res) {
  res.statusCode = 500;
  res.end('Internal Server Error:' + err);
}

var server = http.createServer(function (req, res) {
  // todo: check if the method is POST and type is multipart-data
  if (req.method !== 'POST') {
    res.statusCode = 404;
    return res.end();
  }

  // creating tmp file name
  tmp.file(function(err, path) {
    if (err) return handleError(err, res);

    // uploading file
    debug('Uploading file to ' + path);

    var tmpStream = fs.createWriteStream(path);
    req.on('error', function (err) {
      handleError(err, res);
    });
    req.on('end', function () {
      var cmd = SCRIPT_COMMAND.replace('$image', path);
      debug('Executing script: ' + cmd);
      execScript(cmd, function (err, stdout) {
        if (err) return handleError(err, res);

        debug('Result of script execution: ' + stdout);
        res.end(stdout);
      });
    });

    req.pipe(tmpStream);
  });
});

server.listen(8080, function () {
  debug('Listening on port 8080');
});
