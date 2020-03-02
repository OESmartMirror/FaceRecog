async function EventUpdate() {
	await deleteAllRow();
	await objUpdate();
}

function nowEvent(date) {
	var today = new Date();
	var dd = String(today.getDate()).padStart(2, '0');
	var mm = String(today.getMonth() + 1).padStart(2, '0'); //January is 0!
	var yyyy = today.getFullYear();
}

var events = [
  /*{
    start_date:"",
    end_date:"",
    event_name:"",
	event_location:""
  }*/
]; // start_date:"",end_date:"",event_name:"",event_location:""
function objUpdate() {
  getJSON('https://my-json-server.typicode.com/gartou/db/events',
  function(err, data) {
    if (err !== null) {
      console.log('Something went wrong: ' + err);
    } else {
	  
	  for (i=0;i < data.length && 8; i++) {
		events.push({
			start_date:data[i].start_date,
			end_date:data[i].end_date,
			event_name:data[i].event_name,
			event_location:data[i].event_location
	    });
		newRow('eventTable', parseToTime(events[i].start_date)+"-"+parseToTime(events[i].end_date), events[i].event_name, events[i].event_location);
	  }
    }
  });
}

var newRow = function(tableName, first, second, third) {
  var tableRef = document.getElementById(tableName).getElementsByTagName('tbody')[0];
  var newRow   = tableRef.insertRow(tableRef.rows.length);

  var newCell  = newRow.insertCell(0);
  var newText  = document.createTextNode(first)
  newCell.appendChild(newText);
  
  var newCell  = newRow.insertCell(1);
  var newText  = document.createTextNode(second)
  newCell.appendChild(newText);
  
  var newCell  = newRow.insertCell(2);
  var newText  = document.createTextNode(third)
  
  newCell.appendChild(newText);
}

function parseToTime(date) {
	var res = date.split(';')[1].split(',');
	var hh = res[0]
	var mm = res[1]
	return hh+":"+mm;
}

function parseToDate(date) {
	var res = date.split(';')[0].split(',');
	var yyyy = res[0];
	var mm = res[1];
	var dd = res[2];
	return yyyy+"."+mm+"."+dd;
}

function deleteAllRow() {
	var new_tbody = document.createElement('tbody');
	new_tbody.id = "events"
	var old_tbody = document.getElementById('events');
	old_tbody.parentNode.replaceChild(new_tbody,old_tbody);
}

var getJSON = function(url, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'json';
    xhr.onload = function() {
      var status = xhr.status;
      if (status === 200) {
        callback(null, xhr.response);
      } else {
        callback(status, xhr.response);
      }
    };
    xhr.send();
};