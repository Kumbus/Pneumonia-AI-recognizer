import { HttpClient, HttpEventType, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { DialogComponent } from '../dialog/dialog.component';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.scss']
})
export class MainComponent {

  fileName = '';
  dropzoneHover = false;

  uploadProgress:number = 0;

  constructor(private http: HttpClient, public dialog: MatDialog,private _snackBar: MatSnackBar) {}

  onFileSelected(event: any) {

    const file:File = event.target.files[0];

    if (file) {
        this.uploadFile(file)
    }
  }

  onFileDropped(event: any) {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if(files.length > 0)
    {
      const file = files[0];
      if (file)
      {
        this.uploadFile(file)
      }
    }
  }

  uploadFile = (file: File) => {
    this.uploadProgress = 0;
    this.fileName = file.name;
    const formData = new FormData();
    formData.append("name", file.name)
    formData.append("image", file, file.name);


    this.http.post("http://127.0.0.1:8000/images/", formData, {
      reportProgress: true,
      observe: 'events'
    }).subscribe((event:any) =>
      {
        if (event.type == HttpEventType.UploadProgress)
        {
          console.log(this.uploadProgress)
          this.uploadProgress = Math.round(100 * (event.loaded / event.total));
          if(this.uploadProgress >= 100)
            this._snackBar.open("Image uploaded successfully", "Ok",{
              duration: 1500,
              horizontalPosition: 'center',
              verticalPosition: 'top',
              panelClass: ['mat-snackbar', 'mat-accent']
            })
        }
        else if (event.type === HttpEventType.Response) {
          console.log(this.uploadProgress)
          console.log(event)
          if(event.body.isPneumonia)
          {
            this.dialog.open(DialogComponent, {data: "You have pneumonia, go see a doctor!"})
          }
          else
          {
            this.dialog.open(DialogComponent, {data: "You don't have to worry, you are well :)", height: '20%', width: '25%'})
          }
        }
      });
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
