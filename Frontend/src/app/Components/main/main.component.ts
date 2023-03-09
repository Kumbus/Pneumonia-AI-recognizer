import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { DialogComponent } from '../dialog/dialog.component';

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.scss']
})
export class MainComponent {

  fileName = '';
  dropzoneHover = false;
  constructor(private http: HttpClient, public dialog: MatDialog) {}

  onFileSelected(event: any) {

    const file:File = event.target.files[0];

    if (file) {
        this.fileName = file.name;
        const formData = new FormData();
        formData.append("thumbnail", file);

        const upload$ = this.http.post("http://127.0.0.1:8000/images/", formData);
        upload$.subscribe();
    }
  }

  onFileDropped(event: any) {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      console.log(file)
      this.fileName = file.name;
      const formData = new FormData();
      formData.append("name", file.name)
      formData.append("image", file, file.name);


      this.http.post("http://127.0.0.1:8000/images/", formData).subscribe((response:any) => {
        if(response.isPneumonia)
        {
          this.dialog.open(DialogComponent, {data: "You have pneumonia, go see a doctor!"})
        }
        else
        {
          this.dialog.open(DialogComponent, {data: "You don't have to worry, you are well :)", height: '20%', width: '25%'})
        }
      });

    }

  }

  onDragOver(event: any) {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
    this.dropzoneHover = true;
  }

  onDragLeave(event: any) {
    event.preventDefault();
    event.stopPropagation();
    this.dropzoneHover = false;
  }
}
